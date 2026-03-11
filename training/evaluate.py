# =============================================================================
# RetailGraph — Phase 5: Model Evaluation
# =============================================================================
#
# What this file does, in plain English:
#   1. Loads the fine-tuned Qwen2-VL model from Modal Volume
#   2. Runs it on every example in val.jsonl (678 examples)
#   3. Compares the model's JSON output against the ground truth JSON
#   4. Reports per-field accuracy for all schema fields
#   5. Separately reports visual field accuracy (packaging_type, color, logo)
#   6. Benchmarks latency (seconds per product)
#   7. Saves full results to evaluation/eval_results.json
#   8. Prints a clean summary table
#
# HOW TO RUN:
#
#   Quick sanity check (first 20 examples, ~5 mins):
#     modal run training/evaluate.py --max-samples 20
#
#   Full evaluation (all 678 val examples, ~30 mins on A100):
#     modal run training/evaluate.py
#
#   Download results after:
#     modal volume get retailgraph-models eval_results ./evaluation/
#
# =============================================================================

import os
import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import modal

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retailgraph.evaluate")


# =============================================================================
# SECTION 1 — MODAL SETUP
# Defines the remote container, packages, and Volume mounts.
# Reuses the same base image as training so Modal can cache layers.
# =============================================================================

app = modal.App("retailgraph-evaluate")

# Base layer — torch 2.4 + transformers pinned below 5.x.
# torchvision 0.19 is required by qwen-vl-utils image processing utilities.
# bitsandbytes is NOT included — we load in float16, not 4-bit, so we avoid
# the bitsandbytes/transformers version conflicts entirely.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Pin transformers below 5.x — version 5+ broke Qwen2-VL rope_scaling
        # and bitsandbytes compatibility in our testing.
        "transformers>=4.45.0,<5.0.0",
        "accelerate>=0.34.0",
    )
    .pip_install(
        "Pillow>=10.0.0",
        "einops",
        "qwen-vl-utils",
    )
)

# Modal Volumes — same ones used during training.
# Model weights and val.jsonl are already inside these volumes.
data_volume  = modal.Volume.from_name("retailgraph-data",   create_if_missing=False)
model_volume = modal.Volume.from_name("retailgraph-models", create_if_missing=False)

# Paths inside the Modal container — must match what finetune_qwen.py wrote.
REMOTE_DATA_DIR   = Path("/data")
REMOTE_MODELS_DIR = Path("/models")
REMOTE_MODEL_DIR  = REMOTE_MODELS_DIR / "qwen2vl_retailgraph_v1"
REMOTE_VAL_JSONL  = REMOTE_DATA_DIR   / "training" / "val.jsonl"
REMOTE_IMAGES_DIR = REMOTE_DATA_DIR   / "images"   / "train"
REMOTE_EVAL_DIR   = REMOTE_MODELS_DIR / "eval_results"


# =============================================================================
# SECTION 2 — CONFIGURATION
# Accuracy targets and field groupings — all thresholds in one place.
# If a field falls below its target, we add more training pairs for that field.
# =============================================================================

# Accuracy targets.
TEXT_FIELD_TARGET   = 0.90   # 90% — applies to all text fields
VISUAL_FIELD_TARGET = 0.80   # 80% — applies to packaging_type, color, logo

# Numeric tolerance — within 5% of ground truth = correct.
# Example: truth=9.99, pred=10.20 → 2.1% off → CORRECT
NUMERIC_TOLERANCE = 0.05

# Field groupings — determines which scoring function each field uses.
EXACT_MATCH_FIELDS = ["category", "quantity_unit", "brand", "pack_size"]
NUMERIC_FIELDS     = ["price", "quantity_value"]
LIST_FIELDS        = ["dietary_tags", "allergen_list"]
VISUAL_FIELDS      = ["packaging_type", "packaging_color", "has_brand_logo"]

ALL_FIELDS = EXACT_MATCH_FIELDS + NUMERIC_FIELDS + LIST_FIELDS + VISUAL_FIELDS


# =============================================================================
# SECTION 3 — SCORING FUNCTIONS
# One function per field type. Returns 0.0 (wrong) to 1.0 (correct).
# =============================================================================

def score_exact(pred, truth) -> float:
    """
    Case-insensitive exact match.

    Both None = 1.0  (both correctly absent)
    One None  = 0.0  (one present, one absent = mismatch)
    String    = 1.0 if equal after strip+lower, else 0.0
    """
    if truth is None and pred is None:
        return 1.0
    if truth is None or pred is None:
        return 0.0
    if isinstance(truth, str):
        return 1.0 if str(pred).strip().lower() == str(truth).strip().lower() else 0.0
    return 1.0 if pred == truth else 0.0


def score_numeric(pred, truth, tolerance: float = NUMERIC_TOLERANCE) -> float:
    """
    Numeric match within tolerance %.

    truth=5.99, pred=6.20 → 3.5% off → 1.0 (within 5%)
    truth=5.99, pred=7.00 → 16.9% off → 0.0 (exceeds 5%)
    truth=0,    pred=0    → 1.0 (special case: both zero)
    """
    if truth is None and pred is None:
        return 1.0
    if truth is None or pred is None:
        return 0.0
    try:
        pred_f  = float(pred)
        truth_f = float(truth)
    except (TypeError, ValueError):
        return 0.0
    if truth_f == 0:
        return 1.0 if pred_f == 0 else 0.0
    return 1.0 if abs(pred_f - truth_f) / abs(truth_f) <= tolerance else 0.0


def score_list(pred, truth) -> float:
    """
    Jaccard similarity for list fields (dietary_tags, allergen_list).

    ["organic", "gluten-free"] vs ["organic"]    → 0.5  (1 match / 2 union)
    ["organic", "vegan"]       vs ["organic","vegan"] → 1.0
    [] vs []                                     → 1.0  (both correctly empty)
    """
    truth_set = set(str(t).strip().lower() for t in (truth or []))
    pred_set  = set(str(p).strip().lower() for p in (pred  or []))
    if not truth_set and not pred_set:
        return 1.0
    intersection = truth_set & pred_set
    union        = truth_set | pred_set
    return len(intersection) / len(union) if union else 1.0


def score_field(field_name: str, pred_value, truth_value) -> float:
    """Routes each field to the correct scoring function."""
    if field_name in EXACT_MATCH_FIELDS or field_name in VISUAL_FIELDS:
        return score_exact(pred_value, truth_value)
    elif field_name in NUMERIC_FIELDS:
        return score_numeric(pred_value, truth_value)
    elif field_name in LIST_FIELDS:
        return score_list(pred_value, truth_value)
    else:
        return score_exact(pred_value, truth_value)


# =============================================================================
# SECTION 4 — ROPE SCALING PATCH
# =============================================================================
# WHY THIS EXISTS:
#   Qwen2-VL uses MRoPE (Multimodal Rotary Position Embedding). Every attention
#   layer stores a reference to rope_scaling and reads mrope_section from it
#   during each forward pass:
#
#       self.rope_scaling["mrope_section"]   ← crashes if rope_scaling is None
#
#   When Unsloth merges the LoRA adapter and saves the model, rope_scaling
#   is sometimes stripped from both the config AND from each attention layer's
#   own attribute. Patching only model.config is not enough — the crash happens
#   inside self_attn.forward(), which reads self.rope_scaling directly.
#
#   THE FIX:
#     1. Patch model.config.rope_scaling
#     2. Walk every module in the model and patch any that have a rope_scaling
#        attribute set to None or missing mrope_section
#
#   The values [16, 24, 24] are the exact standard Qwen2-VL 7B defaults from
#   the original Qwen/Qwen2-VL-7B-Instruct config on HuggingFace.
# =============================================================================

# Standard Qwen2-VL 7B MRoPE config — [text_dims, height_dims, width_dims].
QWEN2VL_ROPE_SCALING = {
    "type": "mrope",
    "mrope_section": [16, 24, 24],
}


def patch_rope_scaling(model) -> None:
    """
    Patches rope_scaling on model.config AND on every attention layer.

    Why both?
      - model.config.rope_scaling controls model construction
      - Each Qwen2VLAttention layer has its OWN self.rope_scaling attribute
        that it reads directly during forward(). Patching only the config
        does not update the already-constructed attention layers.

    Called immediately after from_pretrained, before any inference.
    """
    patched_layers = 0

    # Step 1 — patch model.config
    cfg = model.config
    config_needs_patch = (
        cfg.rope_scaling is None
        or not isinstance(cfg.rope_scaling, dict)
        or cfg.rope_scaling.get("mrope_section") is None
    )
    if config_needs_patch:
        cfg.rope_scaling = QWEN2VL_ROPE_SCALING
        log.warning("Patched model.config.rope_scaling (was missing mrope_section).")
    else:
        log.info("model.config.rope_scaling is intact.")

    # Step 2 — patch every attention layer that has a broken rope_scaling
    # model.named_modules() walks the entire model tree recursively.
    for name, module in model.named_modules():
        if hasattr(module, "rope_scaling"):
            rs = module.rope_scaling
            needs_patch = (
                rs is None
                or not isinstance(rs, dict)
                or rs.get("mrope_section") is None
            )
            if needs_patch:
                module.rope_scaling = QWEN2VL_ROPE_SCALING
                patched_layers += 1

    if patched_layers:
        log.warning(
            f"Patched rope_scaling on {patched_layers} attention layer(s). "
            "This is a known Unsloth save/load issue — model quality is unaffected."
        )
    else:
        log.info("All attention layers already have intact rope_scaling.")


# =============================================================================
# SECTION 5 — MODEL LOADING
# =============================================================================

def load_model(model_dir: str):
    """
    Loads the fine-tuned Qwen2-VL model in float16 on GPU.

    Why float16 instead of 4-bit?
      - A100 80GB has 80GB VRAM. Qwen2-VL 7B in float16 needs ~14GB.
        Plenty of headroom — no need to quantize.
      - Avoids bitsandbytes import which conflicts with transformers 4.57.x
        in the Modal container environment.

    Returns:
        (model, processor, device)
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    log.info(f"Loading fine-tuned model from {model_dir}...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,      # 'dtype' replaces deprecated 'torch_dtype'
        device_map="auto",        # automatically places layers across available GPUs
        trust_remote_code=True,   # required for Qwen2-VL custom modeling code
    )

    # CRITICAL: patch rope_scaling immediately after load.
    # If this is skipped, inference will crash with NoneType error.
    patch_rope_scaling(model)

    # Set model to eval mode — disables dropout, ensures deterministic output.
    model.eval()

    processor = Qwen2VLProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    log.info(f"Model loaded on {device} in float16.")

    return model, processor, device


# =============================================================================
# SECTION 6 — INFERENCE
# Runs the model on one val example and returns the parsed JSON output.
# =============================================================================

def run_inference(model, processor, device, messages: list, pil_image=None) -> dict:
    """
    Runs the fine-tuned model on a single val example.

    Args:
        model:      loaded Qwen2-VL model
        processor:  Qwen2VLProcessor (tokenizer + image processor)
        device:     "cuda" or "cpu"
        messages:   list of {role, content} from val.jsonl
                    Includes system, user, assistant — we exclude the assistant
                    message since that is what we are predicting.
        pil_image:  PIL.Image for visual pairs, None for text-only pairs

    Returns:
        dict: parsed JSON prediction, or {} if model output was not valid JSON
    """
    import torch

    # Build the prompt using only system + last user message.
    #
    # WHY NOT messages[:-1]?
    #   Each val.jsonl entry contains few-shot examples (multiple user/assistant
    #   turns) followed by the real product turn at the end. Passing all turns
    #   to the model works but massively inflates the prompt (~3x tokens) and
    #   causes the model to sometimes copy from the few-shot examples instead
    #   of extracting the real product.
    #
    #   The correct approach: extract the system message + the last user message
    #   only. The model was trained on this exact format and needs no few-shot
    #   examples at inference time — that's the whole point of fine-tuning.
    system_msg  = next((m for m in messages if m["role"] == "system"), None)
    last_user   = next((m for m in reversed(messages) if m["role"] == "user"), None)

    if not last_user:
        return {}

    formatted = []
    if system_msg:
        formatted.append({"role": "system", "content": system_msg["content"]})

    if pil_image is not None:
        # Visual pair: inject image into the user turn.
        formatted.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": last_user["content"]},
            ],
        })
    else:
        formatted.append({"role": "user", "content": last_user["content"]})

    # Apply the Qwen2-VL chat template to produce the final prompt string.
    text = processor.apply_chat_template(
        formatted,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize — with or without image depending on pair type.
    if pil_image is not None:
        inputs = processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )

    # Move all tensors to GPU.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response.
    # do_sample=False = greedy decoding → deterministic, reproducible output.
    # temperature=1.0 is ignored when do_sample=False but required to avoid
    # a UserWarning from transformers about conflicting generation config.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
                        do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens — not the input prompt.
    input_length = inputs["input_ids"].shape[1]
    new_tokens   = output_ids[0][input_length:]
    response     = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse the JSON response.
    # Strip markdown code fences if the model wrapped its output in ```json ... ```.
    try:
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response.strip())
    except json.JSONDecodeError:
        log.info(f"RAW OUTPUT: {response[:300]}")
        return {}


# =============================================================================
# SECTION 7 — MAIN EVALUATION FUNCTION (runs on Modal A100)
# =============================================================================

@app.function(
    gpu="A100-80GB",
    image=base_image,
    volumes={
        str(REMOTE_DATA_DIR):   data_volume,
        str(REMOTE_MODELS_DIR): model_volume,
    },
    timeout=7200,   # 2 hours max — full 678 examples takes ~30 mins on A100
)
def run_evaluation(max_samples: int = None):
    """
    Full evaluation pipeline — runs on the Modal A100 GPU.

    Steps:
        1. Load model from Modal Volume (float16, auto device_map)
        2. Patch rope_scaling if missing (Unsloth save/load issue)
        3. Load val.jsonl from Modal Volume
        4. For each example: run inference, score all fields
        5. Save checkpoint every 50 examples
        6. Print + return the final accuracy report

    Args:
        max_samples: limit to N examples for quick test. None = all 678.
    """
    from PIL import Image

    # Load model — rope_scaling patch is applied inside load_model().
    model, processor, device = load_model(str(REMOTE_MODEL_DIR))

    # Load val examples.
    log.info(f"Loading val examples from {REMOTE_VAL_JSONL}...")
    with open(REMOTE_VAL_JSONL, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_samples:
        lines = lines[:max_samples]
        log.info(f"Limited to {max_samples} examples.")

    log.info(f"Evaluating {len(lines)} examples...")

    # Create output directory inside the volume.
    REMOTE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Tracking variables.
    results      = []
    field_scores = defaultdict(list)
    latencies    = []
    text_count   = 0
    visual_count = 0
    parse_errors = 0

    # ── Evaluation loop ───────────────────────────────────────────────────────
    for i, line in enumerate(lines):

        # Parse the JSONL line.
        try:
            pair = json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"Skipping malformed JSONL at line {i+1}")
            continue

        messages   = pair.get("messages", [])
        image_path = pair.get("image_path")

        # Extract ground truth from the LAST assistant message.
        # Each pair has few-shot examples (user/assistant turns) followed by
        # the real product turn. next() would grab the first few-shot assistant
        # message (always the same example product). We want the last one.
        assistant_msg = next(
            (m for m in reversed(messages) if m["role"] == "assistant"), None
        )
        if not assistant_msg:
            continue

        try:
            ground_truth = json.loads(assistant_msg["content"])
        except (json.JSONDecodeError, TypeError):
            log.warning(f"Could not parse ground truth at line {i+1}")
            continue

        # Load image for visual pairs.
        # image_path uses Windows backslashes from generation — normalise here.
        pil_image = None
        if image_path:
            fname     = Path(image_path.replace("\\", "/")).name
            full_path = REMOTE_IMAGES_DIR / fname
            if full_path.exists():
                try:
                    pil_image    = Image.open(full_path).convert("RGB")
                    visual_count += 1
                except Exception as e:
                    log.warning(f"Could not open image {fname}: {e}")
                    text_count += 1
            else:
                log.debug(f"Image not found in volume: {fname} — treating as text-only")
                text_count += 1
        else:
            text_count += 1

        # Run inference and measure latency.
        t0         = time.time()
        prediction = run_inference(model, processor, device, messages, pil_image)
        latency    = time.time() - t0
        latencies.append(latency)

        if not prediction:
            parse_errors += 1

        # Score every field.
        example_scores = {}
        for field in ALL_FIELDS:
            pred_val  = prediction.get(field)
            truth_val = ground_truth.get(field)
            score     = score_field(field, pred_val, truth_val)
            example_scores[field] = score
            field_scores[field].append(score)

        overall = (
            sum(example_scores.values()) / len(example_scores)
            if example_scores else 0.0
        )

        results.append({
            "example_index": i,
            "is_visual":     pil_image is not None,
            "latency_s":     round(latency, 3),
            "overall_score": round(overall, 4),
            "field_scores":  {k: round(v, 4) for k, v in example_scores.items()},
            "prediction":    prediction,
            "ground_truth":  ground_truth,
        })

        # Progress log every 10 examples.
        if (i + 1) % 10 == 0 or (i + 1) == len(lines):
            recent_lat = latencies[-min(10, len(latencies)):]
            avg_lat    = sum(recent_lat) / len(recent_lat)
            log.info(
                f"[{i+1:>3}/{len(lines)}] "
                f"overall={overall:.3f} | "
                f"latency={latency:.1f}s | "
                f"avg_last10={avg_lat:.1f}s"
            )

        # Checkpoint every 50 examples — nothing lost if the run crashes.
        if (i + 1) % 50 == 0:
            _save_results(results, field_scores, latencies,
                          text_count, visual_count, parse_errors)
            model_volume.commit()
            log.info(f"Checkpoint saved at {i+1} examples.")

    # Final save.
    _save_results(results, field_scores, latencies,
                  text_count, visual_count, parse_errors)
    model_volume.commit()

    # Build, print, and return the report.
    report = _build_report(
        field_scores, latencies, text_count, visual_count,
        parse_errors, len(results)
    )
    print(report)
    return report


# =============================================================================
# SECTION 8 — RESULT SAVING AND REPORT GENERATION
# =============================================================================

def _save_results(
    results, field_scores, latencies,
    text_count, visual_count, parse_errors
) -> None:
    """Saves partial or final results to the Modal Volume as JSON."""
    summary = {
        "timestamp":    datetime.now().isoformat(),
        "total":        len(results),
        "text_only":    text_count,
        "visual":       visual_count,
        "parse_errors": parse_errors,
        "avg_latency":  round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "field_accuracy": {
            field: round(sum(scores) / len(scores), 4) if scores else 0.0
            for field, scores in field_scores.items()
        },
        "results": results,
    }
    results_file = REMOTE_EVAL_DIR / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _build_report(
    field_scores, latencies, text_count, visual_count,
    parse_errors, total
) -> str:
    """Formats a clean accuracy report string for terminal output."""
    lines = []
    sep   = "=" * 65

    lines.append(sep)
    lines.append("RETAILGRAPH — PHASE 5 EVALUATION REPORT")
    lines.append(sep)
    lines.append(f"Total examples : {total}")
    lines.append(f"  Text-only    : {text_count}")
    lines.append(f"  Visual pairs : {visual_count}")
    lines.append(f"  Parse errors : {parse_errors}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        lines.append(f"Avg latency    : {avg:.2f}s per example")
    lines.append("")

    lines.append("── TEXT FIELD ACCURACY  (target ≥90%) " + "─" * 26)
    all_passed = True

    for field in EXACT_MATCH_FIELDS + NUMERIC_FIELDS + LIST_FIELDS:
        scores   = field_scores.get(field, [])
        accuracy = sum(scores) / len(scores) if scores else 0.0
        passed   = accuracy >= TEXT_FIELD_TARGET
        status   = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        lines.append(f"  {field:<25} {accuracy*100:5.1f}%   {status}")

    lines.append("")
    lines.append("── VISUAL FIELD ACCURACY (target ≥80%) " + "─" * 25)

    for field in VISUAL_FIELDS:
        scores   = field_scores.get(field, [])
        accuracy = sum(scores) / len(scores) if scores else 0.0
        passed   = accuracy >= VISUAL_FIELD_TARGET
        status   = "✅ PASS" if passed else "❌ FAIL"
        # Only count as failure if there were actually visual examples evaluated
        if not passed and visual_count > 0:
            all_passed = False
        lines.append(f"  {field:<25} {accuracy*100:5.1f}%   {status}")

    lines.append("")
    lines.append("── OVERALL " + "─" * 54)
    all_scores = [s for scores in field_scores.values() for s in scores]
    overall    = sum(all_scores) / len(all_scores) if all_scores else 0.0
    lines.append(f"  Overall accuracy : {overall*100:.1f}%")
    lines.append("")

    if all_passed:
        lines.append("🎉 ALL FIELDS PASSED — ready for Phase 6!")
    else:
        lines.append("⚠️  Some fields below target.")
        lines.append("   Add targeted training pairs for failed fields and retrain.")

    lines.append(sep)
    lines.append("Download results:")
    lines.append("  modal volume get retailgraph-models eval_results ./evaluation/")
    lines.append(sep)

    return "\n".join(lines)


# =============================================================================
# SECTION 9 — LOCAL ENTRYPOINT
# Runs on your laptop and triggers the evaluation on Modal A100.
# =============================================================================

@app.local_entrypoint()
def main(max_samples: int = 0):
    """
    Triggers evaluation on Modal A100 from your laptop.

    Usage:
        # Quick sanity check — 20 examples, ~5 mins, ~$0.15:
        modal run training/evaluate.py --max-samples 20

        # Full evaluation — 678 examples, ~30 mins, ~$2:
        modal run training/evaluate.py

        # Download results after either run:
        modal volume get retailgraph-models eval_results ./evaluation/
    """
    samples = max_samples if max_samples > 0 else None

    if samples:
        log.info(f"Quick test mode: evaluating {samples} examples.")
    else:
        log.info("Full evaluation mode: evaluating all val examples.")

    log.info("Triggering evaluation on Modal A100...")
    log.info("Check progress at: https://modal.com/apps")

    report = run_evaluation.remote(max_samples=samples)

    print("\n" + report)
    log.info("Evaluation complete.")
    log.info("Download results:")
    log.info("  modal volume get retailgraph-models eval_results ./evaluation/")












