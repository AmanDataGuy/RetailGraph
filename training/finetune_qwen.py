# =============================================================================
# RetailGraph — Phase 5: Fine-tuning Qwen2-VL 7B on Seed Data
# =============================================================================
#
# What this file does, in plain English:
#   1. Defines a Modal app that runs on an A100 80GB GPU in the cloud
#   2. Installs all required packages inside that cloud container
#   3. Reads our train.jsonl (3,208 pairs — text + visual + synthetic)
#   4. For visual pairs: loads the product image from Modal Volume
#      For text pairs:   skips image loading entirely
#   5. Fine-tunes Qwen2-VL 7B using QLoRA (trains only small adapter matrices,
#      not all 7 billion parameters)
#   6. Logs every metric to MLflow (loss, learning rate, accuracy per epoch)
#   7. Merges the LoRA adapter back into the base model
#   8. Saves the final model to Modal Volume so you can download it
#
# How to run from your laptop (one command):
#   modal run training/finetune_qwen.py
#
# How to just upload images to Modal Volume (run this first, once):
#   modal run training/finetune_qwen.py::upload_images
#
# =============================================================================

import os
import json
import base64
import logging
from pathlib import Path

import modal

# ── Logging setup ─────────────────────────────────────────────────────────────
# Simple logger so we can see what's happening during training.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retailgraph.finetune")


# =============================================================================
# SECTION 1 — MODAL SETUP
# This section tells Modal what kind of machine to use, what packages to
# install, and where to store our data and model outputs.
# =============================================================================

# The Modal app — think of this as the project name on Modal's platform.
app = modal.App("retailgraph-finetune")

# ── Container image ───────────────────────────────────────────────────────────
# This defines the software environment on the remote GPU machine.
# Modal builds this once and caches it — subsequent runs are instant.
image = (
    modal.Image.debian_slim(python_version="3.11")

    # System packages needed before Python packages.
    .apt_install("git", "wget", "curl")

    # Unsloth — makes QLoRA training 2x faster than vanilla HuggingFace.
    # Must be installed before transformers to override certain kernels.
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )

    # Core ML packages.
    .pip_install(
        "transformers>=4.45.0",   # Qwen2-VL support added in 4.45
        "accelerate>=0.34.0",     # Multi-GPU and mixed precision training
        "peft>=0.12.0",           # LoRA implementation (used by Unsloth)
        "bitsandbytes>=0.43.0",   # 4-bit quantization
        "trl>=0.11.0",            # SFTTrainer (fine-tuning helper)
        "datasets>=2.20.0",       # HuggingFace dataset utilities
    )

    # Tracking and utilities.
    .pip_install(
        "mlflow>=2.16.0",         # Experiment tracking
        "pyyaml>=6.0",            # Reading our YAML config file
        "Pillow>=10.0.0",         # Image loading
        "einops",                 # Required by Qwen2-VL attention
        "qwen-vl-utils",          # Qwen2-VL image preprocessing utilities
    )
)

# ── Modal Volumes ─────────────────────────────────────────────────────────────
# Volumes are persistent storage on Modal — they survive after the GPU
# shuts down. We use two volumes:
#
#   data-volume   → stores our training JSONL files + 500 product images
#   model-volume  → stores checkpoints and the final trained model
#
# Create these once from your terminal:
#   modal volume create retailgraph-data
#   modal volume create retailgraph-models

data_volume  = modal.Volume.from_name("retailgraph-data",   create_if_missing=True)
model_volume = modal.Volume.from_name("retailgraph-models", create_if_missing=True)

# Paths inside the remote container where volumes are mounted.
REMOTE_DATA_DIR   = Path("/data")
REMOTE_MODELS_DIR = Path("/models")

# Where training outputs (checkpoints + final model) go inside the container.
CHECKPOINT_DIR  = REMOTE_MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = REMOTE_MODELS_DIR / "qwen2vl_retailgraph_v1"
MLFLOW_DIR      = REMOTE_MODELS_DIR / "mlruns"


# =============================================================================
# SECTION 2 — IMAGE UPLOAD FUNCTION
# Run this ONCE before training to upload your 500 visual pair images
# and training JSONL files to the Modal data volume.
#
# From your laptop:
#   modal run training/finetune_qwen.py::upload_images
# =============================================================================

@app.function(
    volumes={str(REMOTE_DATA_DIR): data_volume},
    # No GPU needed for just uploading files.
    timeout=3600,
)
def upload_images():
    """
    Uploads the 500 visual pair images + JSONL files to Modal Volume.
    Run this once before running the main fine-tuning function.
    """
    import shutil

    log.info("Starting data upload to Modal Volume...")

    # Create directories inside the volume.
    images_dir = REMOTE_DATA_DIR / "images" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)

    training_dir = REMOTE_DATA_DIR / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    log.info("Directories created inside volume.")

    # NOTE: When Modal mounts local files, they appear at /root/local_data.
    # We copy them into the volume so they persist after this run.
    local_training = Path("/root/local_data/training")
    local_images   = Path("/root/local_data/images/train")

    # Copy JSONL files.
    for fname in ["train.jsonl", "val.jsonl"]:
        src = local_training / fname
        dst = training_dir / fname
        shutil.copy2(src, dst)
        log.info(f"  Copied {fname}")

    # Copy only the 500 visual pair images (not all 75k images).
    # We read visual_pairs.jsonl to get the exact list of image filenames.
    visual_pairs_path = local_training / "visual_pairs.jsonl"
    image_ids = set()

    with open(visual_pairs_path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            img_path = pair.get("image_path", "")
            if img_path:
                # Extract just the filename: data\images\train\150486.jpg → 150486.jpg
                fname = Path(img_path.replace("\\", "/")).name
                image_ids.add(fname)

    log.info(f"Uploading {len(image_ids)} visual pair images...")
    uploaded = 0

    for fname in image_ids:
        src = local_images / fname
        dst = images_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            uploaded += 1
        else:
            log.warning(f"  Image not found locally: {fname}")

    # Commit the volume so changes are saved.
    data_volume.commit()

    log.info(f"Upload complete: {uploaded} images + 2 JSONL files saved to Modal Volume.")


# =============================================================================
# SECTION 3 — DATASET CLASS
# Reads train.jsonl line by line and prepares each example for the model.
#
# Each line in train.jsonl is one training pair:
#   {
#     "messages": [system, user, assistant],
#     "image_path": "data\\images\\train\\150486.jpg"  ← only in visual pairs
#   }
#
# For text-only pairs: image_path is null or missing → no image loaded.
# For visual pairs:    image_path points to an image → load it.
# =============================================================================

def load_dataset_from_jsonl(jsonl_path: str, images_dir: str, processor, max_samples: int = None):
    """
    Reads a JSONL file and returns a list of processed examples.

    Each example is a dict with:
      - input_ids: tokenized text sequence
      - labels:    same as input_ids but with -100 for non-assistant tokens
                   (we only compute loss on the assistant's output)
      - pixel_values: image tensor (only for visual pairs, else None)

    Args:
        jsonl_path:  path to train.jsonl or val.jsonl
        images_dir:  directory where product images are stored
        processor:   Qwen2-VL processor (handles both text tokenization + image preprocessing)
        max_samples: optional limit (useful for quick testing)
    """
    from PIL import Image

    examples = []
    images_dir = Path(images_dir)

    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()

    if max_samples:
        lines = lines[:max_samples]

    text_count   = 0
    visual_count = 0
    skip_count   = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            pair = json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"  Skipping malformed JSON at line {i+1}")
            skip_count += 1
            continue

        messages   = pair.get("messages", [])
        image_path = pair.get("image_path")

        # ── Try to load the image (visual pairs only) ──────────────────────
        pil_image = None
        if image_path:
            # image_path in the JSONL looks like: data\images\train\150486.jpg
            # We just want the filename (150486.jpg) and look it up in images_dir.
            fname     = Path(image_path.replace("\\", "/")).name
            full_path = images_dir / fname

            if full_path.exists():
                try:
                    pil_image = Image.open(full_path).convert("RGB")
                    visual_count += 1
                except Exception as e:
                    log.warning(f"  Could not open image {fname}: {e}")
                    text_count += 1
            else:
                # Image not found on disk — treat as text-only pair.
                # This can happen if an image was missing when we ran generate_visual_pairs.py
                text_count += 1
        else:
            text_count += 1

        # ── Format messages for Qwen2-VL processor ────────────────────────
        # If we have an image, inject it into the user message.
        # Qwen2-VL expects: {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}
        formatted_messages = []

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "user" and pil_image is not None:
                # Visual pair: user message gets image + text
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": content},
                    ],
                })
            else:
                # Text-only: normal message
                formatted_messages.append({
                    "role": role,
                    "content": content,
                })

        # ── Tokenize with processor ────────────────────────────────────────
        # The processor converts text to token IDs and image to pixel values.
        try:
            images_for_processor = [pil_image] if pil_image else None

            inputs = processor(
                text=processor.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
                images=images_for_processor,
                return_tensors="pt",
                padding=False,
            )

            # ── Build labels ──────────────────────────────────────────────
            # Labels = same as input_ids BUT with -100 for every token
            # that is NOT part of the assistant's response.
            # The model only learns from tokens where label != -100.
            # This is how "train on responses only" works.
            input_ids = inputs["input_ids"][0]
            labels    = input_ids.clone()

            # Find where the assistant's response starts.
            # Qwen2-VL uses "<|im_start|>assistant" as the separator.
            assistant_token_id = processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )

            # Walk through input_ids to find the last occurrence of assistant start.
            # Everything before it gets masked with -100.
            assistant_start = None
            for idx in range(len(input_ids) - len(assistant_token_id)):
                if input_ids[idx : idx + len(assistant_token_id)].tolist() == assistant_token_id:
                    assistant_start = idx + len(assistant_token_id)

            if assistant_start is not None:
                labels[:assistant_start] = -100  # Mask system + user tokens
            else:
                # Couldn't find assistant start — mask everything (skip this example)
                labels[:] = -100

            example = {
                "input_ids":      input_ids,
                "attention_mask": inputs["attention_mask"][0],
                "labels":         labels,
            }

            # Only add pixel_values if this is a visual pair.
            if "pixel_values" in inputs:
                example["pixel_values"] = inputs["pixel_values"][0]

            examples.append(example)

        except Exception as e:
            log.warning(f"  Could not process example at line {i+1}: {e}")
            skip_count += 1
            continue

    log.info(
        f"Dataset loaded: {len(examples)} examples "
        f"({text_count} text-only, {visual_count} visual, {skip_count} skipped)"
    )
    return examples


# =============================================================================
# SECTION 4 — DATA COLLATOR
# When the Trainer loads a batch of examples, they might have different lengths
# (one product listing is 50 tokens, another is 200 tokens).
# The collator pads all examples in a batch to the same length so they can
# be stacked into a single tensor for the GPU.
# =============================================================================

class RetailGraphCollator:
    """
    Pads a batch of variable-length examples to the same length.
    Handles both text-only pairs (no pixel_values) and visual pairs.
    """

    def __init__(self, processor, max_seq_length: int = 2048):
        self.processor       = processor
        self.max_seq_length  = max_seq_length
        self.pad_token_id    = processor.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        import torch

        # Find the max sequence length in this batch (capped at max_seq_length).
        max_len = min(
            max(len(ex["input_ids"]) for ex in batch),
            self.max_seq_length,
        )

        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []
        pixel_values_list   = []
        has_images          = False

        for ex in batch:
            ids  = ex["input_ids"][:max_len]
            mask = ex["attention_mask"][:max_len]
            labs = ex["labels"][:max_len]

            # Pad to max_len with pad tokens / -100 for labels.
            pad_len = max_len - len(ids)
            ids  = torch.cat([ids,  torch.full((pad_len,), self.pad_token_id)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), -100)])

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labs)

            if "pixel_values" in ex:
                pixel_values_list.append(ex["pixel_values"])
                has_images = True

        result = {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels":         torch.stack(labels_list),
        }

        if has_images and pixel_values_list:
            # Stack images into a batch tensor.
            # Note: some examples in the batch may not have images.
            # We only include pixel_values if ALL examples in the batch have them.
            # Mixed batches (some with image, some without) are handled by
            # the model internally when pixel_values is None for some examples.
            result["pixel_values"] = torch.stack(pixel_values_list)

        return result


# =============================================================================
# SECTION 5 — MAIN FINE-TUNING FUNCTION
# This is the function that runs on the A100 GPU on Modal.
# Everything above this is setup. This is where the actual training happens.
# =============================================================================

@app.function(
    # Use A100 80GB GPU — enough for Qwen2-VL 7B in 4-bit + LoRA adapters.
    gpu="A100-80GB",

    # The container image we defined in Section 1.
    image=image,

    # Mount our Modal Volumes into the container.
    volumes={
        str(REMOTE_DATA_DIR):   data_volume,
        str(REMOTE_MODELS_DIR): model_volume,
    },

    # No mounts needed — config is read locally and passed as a dict parameter.

    # Maximum runtime: 4 hours. Training should finish in ~2 hours.
    # Modal stops the function if it runs longer than this.
    timeout=14400,

    # Retry once if the function fails (e.g. spot instance preemption).
    retries=1,
)
def finetune(cfg: dict):
    """
    Main fine-tuning function. Runs entirely on the A100 80GB GPU.

    cfg is the parsed qwen2vl_lora.yaml dict — read on your laptop and
    passed directly to this remote function. No file mounting needed.

    Steps:
      1. Unpack config dict
      2. Load Qwen2-VL 7B in 4-bit via Unsloth
      3. Attach LoRA adapters
      4. Load and process train.jsonl + val.jsonl
      5. Run training with HuggingFace Trainer + MLflow logging
      6. Merge LoRA adapter into base model
      7. Save final model to Modal Volume
    """
    import torch
    import mlflow
    from transformers import TrainingArguments, Trainer
    from unsloth import FastVisionModel

    # ── Step 1: Unpack config ─────────────────────────────────────────────────
    # Config was read from qwen2vl_lora.yaml on your laptop and passed here
    # as a plain Python dict. No file I/O needed on the remote machine.
    log.info("Unpacking config...")

    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]
    output_cfg   = cfg["output"]
    mlflow_cfg   = cfg["mlflow"]

    log.info(f"Config loaded. Model: {model_cfg['name']}, LoRA rank: {lora_cfg['r']}")

    # ── Step 2: Load Qwen2-VL 7B in 4-bit via Unsloth ────────────────────────
    # Unsloth's FastVisionModel is a drop-in replacement for HuggingFace's
    # AutoModelForVision2Seq but ~2x faster due to custom CUDA kernels.
    log.info("Loading Qwen2-VL 7B base model in 4-bit...")

    model, processor = FastVisionModel.from_pretrained(
        model_name    = model_cfg["name"],
        load_in_4bit  = model_cfg["load_in_4bit"],
        max_seq_length= model_cfg["max_seq_length"],

        # dtype=None tells Unsloth to auto-detect the best dtype for this GPU.
        # On A100 it will use bfloat16.
        dtype=None,
    )

    log.info(f"Model loaded. Parameters: ~7B total, ~20M trainable (LoRA only)")

    # ── Step 3: Attach LoRA adapters ─────────────────────────────────────────
    # get_peft_model wraps the model with LoRA adapters on the specified layers.
    # After this, model.parameters() only returns the LoRA adapter parameters —
    # the base model weights are frozen (require_grad=False).
    log.info("Attaching LoRA adapters...")

    model = FastVisionModel.get_peft_model(
        model,
        r                   = lora_cfg["r"],
        lora_alpha          = lora_cfg["lora_alpha"],
        lora_dropout        = lora_cfg["lora_dropout"],
        target_modules      = lora_cfg["target_modules"],

        # Apply LoRA to both the language model AND the vision encoder.
        # This is important because we want to fine-tune the visual understanding
        # of packaging types — not just the text extraction.
        finetune_vision_layers    = True,
        finetune_language_layers  = True,
        finetune_attention_layers = True,
        finetune_mlp_layers       = True,

        use_gradient_checkpointing = lora_cfg["use_gradient_checkpointing"],
        random_state               = training_cfg["seed"],
    )

    # Print trainable parameter count so we can verify LoRA is working.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Step 4: Load datasets ─────────────────────────────────────────────────
    # The data lives in our Modal Volume at /data/training/
    train_jsonl  = REMOTE_DATA_DIR / "training" / "train.jsonl"
    val_jsonl    = REMOTE_DATA_DIR / "training" / "val.jsonl"
    images_dir   = REMOTE_DATA_DIR / "images" / "train"

    log.info("Loading training data...")
    train_examples = load_dataset_from_jsonl(
        str(train_jsonl),
        str(images_dir),
        processor,
    )

    log.info("Loading validation data...")
    val_examples = load_dataset_from_jsonl(
        str(val_jsonl),
        str(images_dir),
        processor,
    )

    log.info(f"Train: {len(train_examples)} examples, Val: {len(val_examples)} examples")

    # Wrap in a simple HuggingFace Dataset-compatible class.
    # The Trainer expects something that supports len() and __getitem__().
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    train_dataset = SimpleDataset(train_examples)
    val_dataset   = SimpleDataset(val_examples)

    # ── Step 5: Training arguments ────────────────────────────────────────────
    # TrainingArguments is HuggingFace's way of configuring the training loop.
    # All values come from our YAML config.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir            = str(CHECKPOINT_DIR),
        num_train_epochs      = training_cfg["num_train_epochs"],
        per_device_train_batch_size = training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = training_cfg["gradient_accumulation_steps"],
        learning_rate         = training_cfg["learning_rate"],
        lr_scheduler_type     = training_cfg["lr_scheduler_type"],
        warmup_steps          = training_cfg["warmup_steps"],
        optim                 = training_cfg["optim"],
        bf16                  = training_cfg["bf16"],
        logging_steps         = training_cfg["logging_steps"],
        eval_steps            = training_cfg["eval_steps"],
        save_steps            = training_cfg["save_steps"],
        save_total_limit      = training_cfg["save_total_limit"],
        evaluation_strategy   = "steps",
        save_strategy         = "steps",
        load_best_model_at_end= True,      # Automatically use the best checkpoint
        metric_for_best_model = "eval_loss",
        greater_is_better     = False,     # Lower eval_loss = better
        dataloader_num_workers= data_cfg["dataloader_num_workers"],
        seed                  = training_cfg["seed"],
        report_to             = "mlflow",  # Send metrics to MLflow automatically
        run_name              = mlflow_cfg["run_name"],
        remove_unused_columns = False,     # Keep pixel_values column
    )

    # ── Step 6: MLflow setup ──────────────────────────────────────────────────
    # MLflow tracks every metric so you can compare training runs later.
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        log.info(f"MLflow run started: {run.info.run_id}")

        # Log all hyperparameters from the YAML config so we can
        # reproduce any run exactly.
        mlflow.log_params({
            "model_name":      model_cfg["name"],
            "load_in_4bit":    model_cfg["load_in_4bit"],
            "lora_r":          lora_cfg["r"],
            "lora_alpha":      lora_cfg["lora_alpha"],
            "lora_dropout":    lora_cfg["lora_dropout"],
            "epochs":          training_cfg["num_train_epochs"],
            "batch_size":      training_cfg["per_device_train_batch_size"],
            "grad_accum":      training_cfg["gradient_accumulation_steps"],
            "effective_batch": training_cfg["per_device_train_batch_size"] * training_cfg["gradient_accumulation_steps"],
            "learning_rate":   training_cfg["learning_rate"],
            "train_examples":  len(train_examples),
            "val_examples":    len(val_examples),
        })

        # ── Step 7: Create Trainer and run training ───────────────────────
        collator = RetailGraphCollator(
            processor      = processor,
            max_seq_length = model_cfg["max_seq_length"],
        )

        trainer = Trainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            data_collator   = collator,
        )

        log.info("=" * 60)
        log.info("Starting training...")
        log.info(f"  Epochs:          {training_cfg['num_train_epochs']}")
        log.info(f"  Effective batch: {training_cfg['per_device_train_batch_size'] * training_cfg['gradient_accumulation_steps']}")
        log.info(f"  Learning rate:   {training_cfg['learning_rate']}")
        log.info(f"  Train examples:  {len(train_examples)}")
        log.info("=" * 60)

        train_result = trainer.train()

        log.info("Training complete!")
        log.info(f"  Total steps:     {train_result.global_step}")
        log.info(f"  Training loss:   {train_result.training_loss:.4f}")

        # Log final training metrics to MLflow.
        mlflow.log_metrics({
            "final_train_loss": train_result.training_loss,
            "total_steps":      train_result.global_step,
        })

        # ── Step 8: Merge LoRA adapter into base model ────────────────────
        # The trained model has two parts:
        #   - Base model (frozen, 7B params)
        #   - LoRA adapters (trained, ~20M params)
        #
        # Merging combines them into a single model.
        # Benefits: simpler deployment, faster inference, no PEFT dependency.
        log.info("Merging LoRA adapter into base model...")

        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Unsloth's save_pretrained_merged handles the merge + save in one step.
        # save_method="merged_16bit" saves in float16 (better for inference than 4-bit).
        model.save_pretrained_merged(
            str(FINAL_MODEL_DIR),
            processor,
            save_method="merged_16bit",
        )

        log.info(f"Final model saved to: {FINAL_MODEL_DIR}")

        # Commit the volume so the model is saved permanently.
        model_volume.commit()

        log.info("Model volume committed. Training complete!")
        log.info("=" * 60)
        log.info("NEXT STEPS:")
        log.info("  1. Download model:  modal volume get retailgraph-models qwen2vl_retailgraph_v1 ./outputs/")
        log.info("  2. Download mlruns: modal volume get retailgraph-models mlruns ./mlruns/")
        log.info("  3. View MLflow:     mlflow ui --backend-store-uri ./mlruns")
        log.info("=" * 60)


# =============================================================================
# SECTION 6 — LOCAL ENTRYPOINT
# This is what runs on YOUR LAPTOP when you type: modal run finetune_qwen.py
# It just calls the remote finetune() function on Modal's A100.
# =============================================================================

@app.local_entrypoint()
def main():
    """
    Entry point — called from your laptop.

    Reads qwen2vl_lora.yaml locally, then passes the config as a plain
    Python dict to finetune() running on Modal's A100 GPU.
    No file mounting needed — Modal serializes the dict automatically.

    Usage:
        # First time: upload data to Modal Volume
        modal run training/finetune_qwen.py::upload_images

        # Then: run fine-tuning
        modal run training/finetune_qwen.py

        # After training: download the model
        modal volume get retailgraph-models qwen2vl_retailgraph_v1 ./outputs/
    """
    import yaml

    # Read the YAML config here on your laptop.
    config_path = Path("training/configs/qwen2vl_lora.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Make sure you run this from the RetailGraph project root."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log.info(f"Config loaded from {config_path}")
    log.info(f"  Model:         {cfg['model']['name']}")
    log.info(f"  LoRA rank:     {cfg['lora']['r']}")
    log.info(f"  Epochs:        {cfg['training']['num_train_epochs']}")
    log.info(f"  Learning rate: {cfg['training']['learning_rate']}")
    log.info("")
    log.info("Triggering fine-tuning on Modal A100...")
    log.info("You can close this terminal — training runs in the cloud.")
    log.info("Check progress at: https://modal.com/apps")

    # Pass config dict to remote function — Modal serializes it automatically.
    finetune.remote(cfg)

    log.info("Fine-tuning job submitted successfully.")