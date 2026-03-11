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
#   6. Uses Unsloth's SFTTrainer + UnslothVisionDataCollator — the correct
#      way to fine-tune vision models with Unsloth (avoids image token mismatch)
#   7. Logs every metric to MLflow (loss, learning rate, accuracy per epoch)
#   8. Merges the LoRA adapter back into the base model
#   9. Saves the final model to Modal Volume so you can download it
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
import logging
from pathlib import Path

import modal

# ── Logging setup ─────────────────────────────────────────────────────────────
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

    # Unsloth FIRST — it will install whatever torch version it needs (2.10.0).
    # We let it pick torch freely, then match torchvision to it after.
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )

    # torchvision AFTER unsloth, no version pin — pip auto-matches to torch 2.10.0.
    # Never pin torchvision manually, it must match whatever torch unsloth installed.
    .pip_install(
        "torchvision",
    )

    # Core ML packages.
    .pip_install(
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "trl>=0.11.0",
        "datasets>=2.20.0",
    )

    # Tracking and utilities.
    .pip_install(
        "mlflow>=2.16.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0",
        "einops",
        "qwen-vl-utils",
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
    timeout=3600,
)
def upload_images():
    """
    Uploads the 500 visual pair images + JSONL files to Modal Volume.
    Run this once before running the main fine-tuning function.
    """
    import shutil

    log.info("Starting data upload to Modal Volume...")

    images_dir = REMOTE_DATA_DIR / "images" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)

    training_dir = REMOTE_DATA_DIR / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    log.info("Directories created inside volume.")

    local_training = Path("/root/local_data/training")
    local_images   = Path("/root/local_data/images/train")

    for fname in ["train.jsonl", "val.jsonl"]:
        src = local_training / fname
        dst = training_dir / fname
        shutil.copy2(src, dst)
        log.info(f"  Copied {fname}")

    visual_pairs_path = local_training / "visual_pairs.jsonl"
    image_ids = set()

    with open(visual_pairs_path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            img_path = pair.get("image_path", "")
            if img_path:
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

    data_volume.commit()

    log.info(f"Upload complete: {uploaded} images + 2 JSONL files saved to Modal Volume.")


# =============================================================================
# SECTION 3 — DATASET LOADING
# Reads train.jsonl line by line and returns a list of dicts ready for
# Unsloth's SFTTrainer.
#
# KEY DIFFERENCE FROM BEFORE:
#   We do NOT pre-tokenize here. We just load the raw messages + PIL image.
#   Unsloth's UnslothVisionDataCollator handles tokenization at batch time,
#   which is the only correct way to avoid image token/feature mismatches.
#
# Each returned dict has:
#   {
#     "messages": [system_msg, user_msg, assistant_msg],
#     # user_msg content is a list: [{"type": "image", "image": PIL_image},
#     #                               {"type": "text",  "text": "..."}]
#     # for visual pairs, or just a string for text-only pairs
#   }
# =============================================================================

def load_dataset_from_jsonl(jsonl_path: str, images_dir: str, max_samples: int = None):
    """
    Reads a JSONL file and returns a list of raw examples for SFTTrainer.

    Does NOT tokenize — just loads messages and PIL images.
    Unsloth's UnslothVisionDataCollator tokenizes at batch time.

    Args:
        jsonl_path:  path to train.jsonl or val.jsonl
        images_dir:  directory where product images are stored
        max_samples: optional limit (useful for quick testing)
    """
    from PIL import Image

    examples    = []
    images_dir  = Path(images_dir)
    text_count  = 0
    visual_count = 0
    skip_count  = 0

    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()

    if max_samples:
        lines = lines[:max_samples]

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
                text_count += 1
        else:
            text_count += 1

        # ── Build the messages list for SFTTrainer ─────────────────────────
        # Unsloth's UnslothVisionDataCollator expects messages in this format:
        #
        # Text-only:
        #   {"role": "user", "content": "some text"}
        #
        # Visual:
        #   {"role": "user", "content": [
        #       {"type": "image", "image": <PIL Image>},
        #       {"type": "text",  "text": "some text"}
        #   ]}
        #
        # The collator calls processor.apply_chat_template internally,
        # so we never touch tokenization ourselves.
        formatted_messages = []

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "user" and pil_image is not None:
                # Visual pair: inject the PIL image into the user message.
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text",  "text": content},
                    ],
                })
            else:
                # Text-only message: content stays as a plain string.
                formatted_messages.append({
                    "role":    role,
                    "content": content,
                })

        examples.append({"messages": formatted_messages})

    log.info(
        f"Dataset loaded: {len(examples)} examples "
        f"({text_count} text-only, {visual_count} visual, {skip_count} skipped)"
    )
    return examples


# =============================================================================
# SECTION 4 — MAIN FINE-TUNING FUNCTION
# This is the function that runs on the A100 GPU on Modal.
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

    # Maximum runtime: 4 hours. Training should finish in ~2 hours.
    timeout=14400,

    # Retry once if the function fails (e.g. spot instance preemption).
    retries=1,
)
def finetune(cfg: dict):
    """
    Main fine-tuning function. Runs entirely on the A100 80GB GPU.

    Uses Unsloth's SFTTrainer + UnslothVisionDataCollator — the correct
    way to fine-tune Qwen2-VL. This avoids the "image features and image
    tokens do not match" error that happens with raw HuggingFace Trainer.

    Steps:
      1. Unpack config dict
      2. Load Qwen2-VL 7B in 4-bit via Unsloth
      3. Attach LoRA adapters
      4. Load train.jsonl + val.jsonl (raw messages, no pre-tokenization)
      5. Run training with SFTTrainer + UnslothVisionDataCollator
      6. Log metrics to MLflow
      7. Merge LoRA adapter into base model
      8. Save final model to Modal Volume
    """
    import torch
    import mlflow
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # ── Step 1: Unpack config ─────────────────────────────────────────────────
    log.info("Unpacking config...")

    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]
    mlflow_cfg   = cfg["mlflow"]

    log.info(f"Config loaded. Model: {model_cfg['name']}, LoRA rank: {lora_cfg['r']}")

    # ── Step 2: Load Qwen2-VL 7B in 4-bit via Unsloth ────────────────────────
    log.info("Loading Qwen2-VL 7B base model in 4-bit...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name    = model_cfg["name"],
        load_in_4bit  = model_cfg["load_in_4bit"],
        max_seq_length= model_cfg["max_seq_length"],
        dtype         = None,  # auto-detect: bfloat16 on A100
    )

    log.info("Model loaded.")

    # ── Step 3: Attach LoRA adapters ─────────────────────────────────────────
    log.info("Attaching LoRA adapters...")

    model = FastVisionModel.get_peft_model(
        model,
        r                          = lora_cfg["r"],
        lora_alpha                 = lora_cfg["lora_alpha"],
        lora_dropout               = lora_cfg["lora_dropout"],
        target_modules             = lora_cfg["target_modules"],
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_layers  = True,
        finetune_mlp_layers        = True,
        use_gradient_checkpointing = lora_cfg["use_gradient_checkpointing"],
        random_state               = training_cfg["seed"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Enable training mode — Unsloth requires this for vision models.
    FastVisionModel.for_training(model)

    # ── Step 4: Load datasets ─────────────────────────────────────────────────
    train_jsonl = REMOTE_DATA_DIR / "training" / "train_r3.jsonl"
    val_jsonl   = REMOTE_DATA_DIR / "training" / "val.jsonl"
    images_dir  = REMOTE_DATA_DIR / "images" / "train"

    log.info("Loading training data...")
    train_examples = load_dataset_from_jsonl(
        str(train_jsonl),
        str(images_dir),
    )

    log.info("Loading validation data...")
    val_examples = load_dataset_from_jsonl(
        str(val_jsonl),
        str(images_dir),
    )

    log.info(f"Train: {len(train_examples)} examples, Val: {len(val_examples)} examples")

    # Wrap in a simple Dataset class that supports len() and __getitem__().
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    train_dataset = SimpleDataset(train_examples)
    val_dataset   = SimpleDataset(val_examples)

    # ── Step 5: MLflow setup ──────────────────────────────────────────────────
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        log.info(f"MLflow run started: {run.info.run_id}")

        mlflow.log_params({
            "model_name":      model_cfg["name"],
            "lora_r":          lora_cfg["r"],
            "lora_alpha":      lora_cfg["lora_alpha"],
            "epochs":          training_cfg["num_train_epochs"],
            "batch_size":      training_cfg["per_device_train_batch_size"],
            "grad_accum":      training_cfg["gradient_accumulation_steps"],
            "learning_rate":   training_cfg["learning_rate"],
            "train_examples":  len(train_examples),
            "val_examples":    len(val_examples),
        })

        # ── Step 6: Create SFTTrainer with UnslothVisionDataCollator ──────────
        # This is the key fix vs the previous version.
        #
        # UnslothVisionDataCollator:
        #   - Takes raw messages + PIL images
        #   - Calls processor.apply_chat_template + processor() internally
        #   - Builds input_ids, pixel_values, image_grid_thw all at once
        #   - Guarantees image tokens match image features (no mismatch)
        #
        # SFTConfig with skip_prepare_dataset=True:
        #   - Tells SFTTrainer not to pre-tokenize — let the collator do it
        #   - Required when using UnslothVisionDataCollator

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
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
            eval_strategy         = "no",
            save_strategy         = "steps",
            load_best_model_at_end= False,
            seed                  = training_cfg["seed"],
            report_to             = "mlflow",
            run_name              = mlflow_cfg["run_name"],
            remove_unused_columns = False,
            # CRITICAL: tell SFTTrainer not to pre-tokenize the dataset.
            # The collator handles tokenization at batch time instead.
            dataset_text_field    = "",
            dataset_kwargs        = {"skip_prepare_dataset": True},
            packing               = False,
            dataloader_drop_last   = True,
            max_seq_length        = model_cfg["max_seq_length"],
            dataloader_num_workers= data_cfg["dataloader_num_workers"],
        )

        trainer = SFTTrainer(
            model         = model,
            tokenizer     = tokenizer,
            args          = sft_config,
            train_dataset = train_dataset,
            data_collator = UnslothVisionDataCollator(model, tokenizer),
        )

        log.info("=" * 60)
        log.info("Starting training...")
        log.info(f"  Epochs:          {training_cfg['num_train_epochs']}")
        log.info(f"  Effective batch: {training_cfg['per_device_train_batch_size'] * training_cfg['gradient_accumulation_steps']}")
        log.info(f"  Learning rate:   {training_cfg['learning_rate']}")
        log.info(f"  Train examples:  {len(train_examples)}")
        log.info("=" * 60)

                # Auto-resume from last checkpoint if one exists
        import os
        last_ckpt = None
        if CHECKPOINT_DIR.exists():
            checkpoints = sorted([d for d in os.listdir(str(CHECKPOINT_DIR)) if d.startswith("checkpoint")])
            if checkpoints:
                last_ckpt = str(CHECKPOINT_DIR / checkpoints[-1])
                log.info(f"Resuming from checkpoint: {last_ckpt}")
        train_result = trainer.train(resume_from_checkpoint=last_ckpt)

        log.info("Training complete!")
        log.info(f"  Total steps:     {train_result.global_step}")
        log.info(f"  Training loss:   {train_result.training_loss:.4f}")

        mlflow.log_metrics({
            "final_train_loss": train_result.training_loss,
            "total_steps":      train_result.global_step,
        })

        # ── Step 7: Merge LoRA adapter into base model ────────────────────────
        # Switch back to inference mode before merging.
        FastVisionModel.for_inference(model)

        log.info("Merging LoRA adapter into base model...")

        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model.save_pretrained_merged(
            str(FINAL_MODEL_DIR),
            tokenizer,
            save_method="merged_16bit",
        )

        log.info(f"Final model saved to: {FINAL_MODEL_DIR}")

        model_volume.commit()

        log.info("Model volume committed. Training complete!")
        log.info("=" * 60)
        log.info("NEXT STEPS:")
        log.info("  1. Download model:  modal volume get retailgraph-models qwen2vl_retailgraph_v1 ./outputs/")
        log.info("  2. Download mlruns: modal volume get retailgraph-models mlruns ./mlruns/")
        log.info("  3. View MLflow:     mlflow ui --backend-store-uri ./mlruns")
        log.info("=" * 60)


# =============================================================================
# SECTION 5 — LOCAL ENTRYPOINT
# This runs on YOUR LAPTOP when you type: modal run finetune_qwen.py
# It reads the YAML config and passes it to finetune() on Modal's A100.
# =============================================================================

@app.local_entrypoint()
def main():
    """
    Entry point — called from your laptop.

    Reads qwen2vl_lora.yaml locally, then passes the config as a plain
    Python dict to finetune() running on Modal's A100 GPU.

    Usage:
        modal run training/finetune_qwen.py::upload_images  # first time only
        modal run training/finetune_qwen.py                 # run training
        modal volume get retailgraph-models qwen2vl_retailgraph_v1 ./outputs/
    """
    import yaml

    config_path = Path("training/configs/qwen2vl_lora.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Make sure you run this from the RetailGraph project root."
        )

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log.info(f"Config loaded from {config_path}")
    log.info(f"  Model:         {cfg['model']['name']}")
    log.info(f"  LoRA rank:     {cfg['lora']['r']}")
    log.info(f"  Epochs:        {cfg['training']['num_train_epochs']}")
    log.info(f"  Learning rate: {cfg['training']['learning_rate']}")
    log.info("")
    log.info("Triggering fine-tuning on Modal A100...")

    finetune.remote(cfg)

    log.info("Fine-tuning job submitted successfully.")


