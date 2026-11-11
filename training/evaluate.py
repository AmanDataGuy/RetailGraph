"""# evaluate.py - scores the fine-tuned Qwen2-VL on val.jsonl, per field, on a Modal A100

- Input: Modal volumes retailgraph-models (/models/qwen2vl_retailgraph_v1) and retailgraph-data (/data/training/val.jsonl, images).
- Output: /models/eval_results/eval_results.json (checkpointed every 50 examples) and a printed pass/fail report.
- Run: modal run training/evaluate.py [--max-samples 20]
  then: modal volume get retailgraph-models eval_results ./evaluation/
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retailgraph.evaluate")


app = modal.App("retailgraph-evaluate")

# No bitsandbytes: the model loads in float16, which avoids bitsandbytes/transformers version clashes.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # transformers 5.x broke Qwen2-VL rope_scaling in our testing.
        "transformers>=4.45.0,<5.0.0",
        "accelerate>=0.34.0",
    )
    .pip_install(
        "Pillow>=10.0.0",
        "einops",
        "qwen-vl-utils",
    )
)

# Same volumes as training; paths must match what finetune_qwen.py wrote.
data_volume  = modal.Volume.from_name("retailgraph-data",   create_if_missing=False)
model_volume = modal.Volume.from_name("retailgraph-models", create_if_missing=False)

REMOTE_DATA_DIR   = Path("/data")
REMOTE_MODELS_DIR = Path("/models")
REMOTE_MODEL_DIR  = REMOTE_MODELS_DIR / "qwen2vl_retailgraph_v1"
REMOTE_VAL_JSONL  = REMOTE_DATA_DIR   / "training" / "val.jsonl"
REMOTE_IMAGES_DIR = REMOTE_DATA_DIR   / "images"   / "train"
REMOTE_EVAL_DIR   = REMOTE_MODELS_DIR / "eval_results"


TEXT_FIELD_TARGET   = 0.90   # 90% — applies to all text fields
VISUAL_FIELD_TARGET = 0.80   # 80% — applies to packaging_type, color, logo

NUMERIC_TOLERANCE = 0.05   # within 5% of the truth counts as correct

# Field groups pick the scoring function.
EXACT_MATCH_FIELDS = ["category", "quantity_unit", "brand", "pack_size"]
NUMERIC_FIELDS     = ["price", "quantity_value"]
LIST_FIELDS        = ["dietary_tags", "allergen_list"]
VISUAL_FIELDS      = ["packaging_type", "packaging_color", "has_brand_logo"]

ALL_FIELDS = EXACT_MATCH_FIELDS + NUMERIC_FIELDS + LIST_FIELDS + VISUAL_FIELDS


# Case-insensitive exact match; both None counts as correct, one None as wrong.
def score_exact(pred, truth) -> float:
    if truth is None and pred is None:
        return 1.0
    if truth is None or pred is None:
        return 0.0
    if isinstance(truth, str):
        return 1.0 if str(pred).strip().lower() == str(truth).strip().lower() else 0.0
    return 1.0 if pred == truth else 0.0


# 1.0 if pred is within `tolerance` (relative) of truth, else 0.0.
def score_numeric(pred, truth, tolerance: float = NUMERIC_TOLERANCE) -> float:
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


# Jaccard similarity of two lists (case-insensitive); two empty lists score 1.0.
def score_list(pred, truth) -> float:
    truth_set = set(str(t).strip().lower() for t in (truth or []))
    pred_set  = set(str(p).strip().lower() for p in (pred  or []))
    if not truth_set and not pred_set:
        return 1.0
    intersection = truth_set & pred_set
    union        = truth_set | pred_set
    return len(intersection) / len(union) if union else 1.0


# Route a field to its scoring function.
def score_field(field_name: str, pred_value, truth_value) -> float:
    if field_name in EXACT_MATCH_FIELDS or field_name in VISUAL_FIELDS:
        return score_exact(pred_value, truth_value)
    elif field_name in NUMERIC_FIELDS:
        return score_numeric(pred_value, truth_value)
    elif field_name in LIST_FIELDS:
        return score_list(pred_value, truth_value)
    else:
        return score_exact(pred_value, truth_value)


# Unsloth's merged save sometimes drops rope_scaling from the config AND from each attention layer,
# and Qwen2-VL attention reads self.rope_scaling["mrope_section"] in forward(). Defaults from Qwen2-VL-7B-Instruct.
QWEN2VL_ROPE_SCALING = {
    "type": "mrope",
    "mrope_section": [16, 24, 24],
}


# Restore the MRoPE rope_scaling on model.config and on every layer that lost it.
def patch_rope_scaling(model) -> None:
    patched_layers = 0

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

    for _name, module in model.named_modules():
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


# Load the merged model in float16 (fits easily on an A100, no bitsandbytes); returns (model, processor, device).
def load_model(model_dir: str):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    log.info(f"Loading fine-tuned model from {model_dir}...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    patch_rope_scaling(model)   # must run before any inference, or forward() crashes on None
    model.eval()

    processor = Qwen2VLProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    log.info(f"Model loaded on {device} in float16.")

    return model, processor, device


# Generate a prediction for one val example; returns the parsed JSON, or {} if it is not valid JSON.
def run_inference(model, processor, device, messages: list, pil_image=None) -> dict:
    import torch

    # Prompt = system + last user turn only; dropping the few-shot turns keeps the prompt ~3x smaller
    # and stops the model copying from the examples.
    system_msg  = next((m for m in messages if m["role"] == "system"), None)
    last_user   = next((m for m in reversed(messages) if m["role"] == "user"), None)

    if not last_user:
        return {}

    formatted = []
    if system_msg:
        formatted.append({"role": "system", "content": system_msg["content"]})

    if pil_image is not None:
        formatted.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": last_user["content"]},
            ],
        })
    else:
        formatted.append({"role": "user", "content": last_user["content"]})

    text = processor.apply_chat_template(
        formatted,
        tokenize=False,
        add_generation_prompt=True,
    )

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

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Greedy decoding; temperature=1.0 only silences a transformers generation-config warning.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens   = output_ids[0][input_length:]
    response     = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip ```json fences if the model added them.
    try:
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response.strip())
    except json.JSONDecodeError:
        log.info(f"RAW OUTPUT: {response[:300]}")
        return {}


# Run the model over val.jsonl (or the first max_samples), score every field, save and return the report.
@app.function(
    gpu="A100-80GB",
    image=base_image,
    volumes={
        str(REMOTE_DATA_DIR):   data_volume,
        str(REMOTE_MODELS_DIR): model_volume,
    },
    timeout=7200,
)
def run_evaluation(max_samples: int = None):
    from PIL import Image

    model, processor, device = load_model(str(REMOTE_MODEL_DIR))

    log.info(f"Loading val examples from {REMOTE_VAL_JSONL}...")
    with open(REMOTE_VAL_JSONL, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_samples:
        lines = lines[:max_samples]
        log.info(f"Limited to {max_samples} examples.")

    log.info(f"Evaluating {len(lines)} examples...")

    REMOTE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    results      = []
    field_scores = defaultdict(list)
    latencies    = []
    text_count   = 0
    visual_count = 0
    parse_errors = 0

    for i, line in enumerate(lines):
        try:
            pair = json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"Skipping malformed JSONL at line {i+1}")
            continue

        messages   = pair.get("messages", [])
        image_path = pair.get("image_path")

        # Ground truth is the LAST assistant turn; earlier ones are few-shot examples.
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

        # image_path was written on Windows, so normalise backslashes.
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

        t0         = time.time()
        prediction = run_inference(model, processor, device, messages, pil_image)
        latency    = time.time() - t0
        latencies.append(latency)

        if not prediction:
            parse_errors += 1

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

        if (i + 1) % 10 == 0 or (i + 1) == len(lines):
            recent_lat = latencies[-min(10, len(latencies)):]
            avg_lat    = sum(recent_lat) / len(recent_lat)
            log.info(
                f"[{i+1:>3}/{len(lines)}] "
                f"overall={overall:.3f} | "
                f"latency={latency:.1f}s | "
                f"avg_last10={avg_lat:.1f}s"
            )

        # Checkpoint so a crash loses at most 50 examples.
        if (i + 1) % 50 == 0:
            _save_results(results, field_scores, latencies,
                          text_count, visual_count, parse_errors)
            model_volume.commit()
            log.info(f"Checkpoint saved at {i+1} examples.")

    _save_results(results, field_scores, latencies,
                  text_count, visual_count, parse_errors)
    model_volume.commit()

    report = _build_report(
        field_scores, latencies, text_count, visual_count,
        parse_errors, len(results)
    )
    print(report)
    return report


# Write the partial or final results JSON to the model volume.
def _save_results(
    results, field_scores, latencies,
    text_count, visual_count, parse_errors
) -> None:
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


# Format the per-field pass/fail report for the terminal.
def _build_report(
    field_scores, latencies, text_count, visual_count,
    parse_errors, total
) -> str:
    lines = []
    sep   = "=" * 65

    lines.append(sep)
    lines.append("RETAILGRAPH — EVALUATION REPORT")
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
        # Visual fields only fail the run if visual examples were actually scored.
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
        lines.append("🎉 ALL FIELDS PASSED")
    else:
        lines.append("⚠️  Some fields below target.")
        lines.append("   Add targeted training pairs for failed fields and retrain.")

    lines.append(sep)
    lines.append("Download results:")
    lines.append("  modal volume get retailgraph-models eval_results ./evaluation/")
    lines.append(sep)

    return "\n".join(lines)


# Local entrypoint: start run_evaluation on Modal and print its report (0 = all examples).
@app.local_entrypoint()
def main(max_samples: int = 0):
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
