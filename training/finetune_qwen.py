"""# finetune_qwen.py - QLoRA fine-tune of Qwen2-VL-7B on the seed pairs, on a Modal A100-80GB

- Input: training/configs/qwen2vl_lora.yaml (read locally) + /data/training/train_r3.jsonl and images on the retailgraph-data volume.
- Output: merged 16-bit model at /models/qwen2vl_retailgraph_v1, checkpoints and MLflow runs on retailgraph-models.
- Run: modal run training/finetune_qwen.py::upload_images  (once), then modal run training/finetune_qwen.py
"""

import json
import logging
from pathlib import Path

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retailgraph.finetune")


app = modal.App("retailgraph-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    # Unsloth first so it picks torch; torchvision stays unpinned so pip matches that torch.
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
    .pip_install(
        "torchvision",
    )
    .pip_install(
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "trl>=0.11.0",
        "datasets>=2.20.0",
    )
    .pip_install(
        "mlflow>=2.16.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0",
        "einops",
        "qwen-vl-utils",
    )
)

# data volume: JSONL files + product images; model volume: checkpoints, final model, mlruns.
data_volume  = modal.Volume.from_name("retailgraph-data",   create_if_missing=True)
model_volume = modal.Volume.from_name("retailgraph-models", create_if_missing=True)

REMOTE_DATA_DIR   = Path("/data")
REMOTE_MODELS_DIR = Path("/models")

CHECKPOINT_DIR  = REMOTE_MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = REMOTE_MODELS_DIR / "qwen2vl_retailgraph_v1"
MLFLOW_DIR      = REMOTE_MODELS_DIR / "mlruns"


# One-off: copy train/val JSONL and the visual-pair images into the data volume.
@app.function(
    volumes={str(REMOTE_DATA_DIR): data_volume},
    timeout=3600,
)
def upload_images():
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


# Load JSONL pairs as raw messages (+ PIL image for visual pairs); UnslothVisionDataCollator tokenizes
# at batch time, which avoids image token/feature mismatches.
def load_dataset_from_jsonl(jsonl_path: str, images_dir: str, max_samples: int = None):
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

        # Visual pairs: user content becomes [{"type": "image"}, {"type": "text"}]; others stay plain strings.
        formatted_messages = []

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "user" and pil_image is not None:
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text",  "text": content},
                    ],
                })
            else:
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


# QLoRA fine-tune on the A100: load 4-bit, attach LoRA, train with SFTTrainer, merge and save to the volume.
@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={
        str(REMOTE_DATA_DIR):   data_volume,
        str(REMOTE_MODELS_DIR): model_volume,
    },
    timeout=14400,
    retries=1,
)
def finetune(cfg: dict):
    import torch
    import mlflow
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    log.info("Unpacking config...")

    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]
    mlflow_cfg   = cfg["mlflow"]

    log.info(f"Config loaded. Model: {model_cfg['name']}, LoRA rank: {lora_cfg['r']}")

    log.info("Loading Qwen2-VL 7B base model in 4-bit...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name    = model_cfg["name"],
        load_in_4bit  = model_cfg["load_in_4bit"],
        max_seq_length= model_cfg["max_seq_length"],
        dtype         = None,  # auto-detect: bfloat16 on A100
    )

    log.info("Model loaded.")

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

    # Unsloth requires this for vision models.
    FastVisionModel.for_training(model)

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

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    train_dataset = SimpleDataset(train_examples)

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
            # Do not pre-tokenize: UnslothVisionDataCollator does it per batch.
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

        # Auto-resume from the last checkpoint if one exists.
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


# Local entrypoint: read the YAML config on this machine and start finetune() on Modal.
@app.local_entrypoint()
def main():
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
