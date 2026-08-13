"""
Main training script — trains a Seq2SeqTransformer (C1–C4) or BLTSeq2SeqModel (C5)
on the cipher decryption task.

CLI flags control the ablation axis:
    C1 (base):  all defaults
    C2:         --positional_encoding rope
    C3:         --attention_type gqa
    C4:         --norm_type rmsnorm
    C5:         --tokenization blt

All shared hyperparameters are pulled from a single defaults dict to ensure
they're identical across all five configurations.

Usage:
    python src/train.py --run_name c1
    python src/train.py --run_name c2 --positional_encoding rope
    python src/train.py --run_name c3 --attention_type gqa
    python src/train.py --run_name c4 --norm_type rmsnorm
    python src/train.py --run_name c5 --tokenization blt
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import build_dataloaders, BYTE_PAD, BYTE_BOS, BYTE_EOS, PLAIN_CHUNK_SIZE
from src.models import Seq2SeqTransformer
from src.models.blt import BLTSeq2SeqModel, BYTE_VOCAB_SIZE
from src.utils import (
    init_wandb, log_wandb, finish_wandb,
    save_and_push, push_folder_to_hub, push_to_hub,
    compute_all_metrics, save_metrics_json,
)


# ═══════════════════════════════════════════════════════════════════════
# Shared Defaults (identical across C1–C5)
# ═══════════════════════════════════════════════════════════════════════

SHARED_DEFAULTS = {
    "d_model": 256,
    "num_heads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "learning_rate": 3e-4,
    "warmup_steps": 2000,
    "batch_size": 128,
    "grad_clip_norm": 1.0,
    "max_seq_len": 150,
    "seed": 42,
    "epochs": 25,
    "patience": 5,
    "bpe_vocab_size": 8000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cipher Transformer Ablation Training")

    # Ablation flags
    parser.add_argument("--positional_encoding", choices=["sinusoidal", "rope"],
                        default="sinusoidal", help="Positional encoding type (C2: rope)")
    parser.add_argument("--attention_type", choices=["mha", "gqa"],
                        default="mha", help="Attention type (C3: gqa)")
    parser.add_argument("--norm_type", choices=["layernorm", "rmsnorm"],
                        default="layernorm", help="Normalization type (C4: rmsnorm)")
    parser.add_argument("--tokenization", choices=["subword", "blt"],
                        default="subword", help="Tokenization mode (C5: blt)")
    parser.add_argument("--gqa_kv_heads", type=int, default=2,
                        help="KV heads for GQA (used only if attention_type=gqa)")
    parser.add_argument("--blt_patch_size", type=int, default=9,
                        help="Patch size for BLT (used only if tokenization=blt)")

    # Run identification
    parser.add_argument("--run_name", type=str, default="c1",
                        help="W&B run name, checkpoint dir name, HF repo suffix")

    # W&B and HF
    parser.add_argument("--wandb_project", type=str, default="ANLP_A1",
                        help="W&B project name")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                        help="HF repo ID (default: <hf_user>/cipher-transformer-<run_name>)")

    # Overrides (normally use SHARED_DEFAULTS)
    parser.add_argument("--d_model", type=int, default=SHARED_DEFAULTS["d_model"])
    parser.add_argument("--num_heads", type=int, default=SHARED_DEFAULTS["num_heads"])
    parser.add_argument("--num_encoder_layers", type=int, default=SHARED_DEFAULTS["num_encoder_layers"])
    parser.add_argument("--num_decoder_layers", type=int, default=SHARED_DEFAULTS["num_decoder_layers"])
    parser.add_argument("--d_ff", type=int, default=SHARED_DEFAULTS["d_ff"])
    parser.add_argument("--dropout", type=float, default=SHARED_DEFAULTS["dropout"])
    parser.add_argument("--label_smoothing", type=float, default=SHARED_DEFAULTS["label_smoothing"])
    parser.add_argument("--learning_rate", type=float, default=SHARED_DEFAULTS["learning_rate"])
    parser.add_argument("--warmup_steps", type=int, default=SHARED_DEFAULTS["warmup_steps"])
    parser.add_argument("--batch_size", type=int, default=SHARED_DEFAULTS["batch_size"])
    parser.add_argument("--grad_clip_norm", type=float, default=SHARED_DEFAULTS["grad_clip_norm"])
    parser.add_argument("--max_seq_len", type=int, default=SHARED_DEFAULTS["max_seq_len"])
    parser.add_argument("--seed", type=int, default=SHARED_DEFAULTS["seed"])
    parser.add_argument("--epochs", type=int, default=SHARED_DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=SHARED_DEFAULTS["patience"])
    parser.add_argument("--bpe_vocab_size", type=int, default=SHARED_DEFAULTS["bpe_vocab_size"])

    # Hardware
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--num_workers", type=int, default=0)

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Max samples for greedy decode during validation (speed)")

    return parser.parse_args()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay schedule."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def build_model(args, data_info: dict, device: torch.device) -> nn.Module:
    """Build the appropriate model based on config flags."""
    if args.tokenization == "blt":
        model = BLTSeq2SeqModel(
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            patch_size=args.blt_patch_size,
            d_local=args.d_model // 2,
            local_heads=max(1, args.num_heads // 2),
        )
    else:
        model = Seq2SeqTransformer(
            src_vocab_size=data_info["src_vocab_size"],
            tgt_vocab_size=data_info["tgt_vocab_size"],
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            pad_idx=data_info["pad_idx"],
            attention_type=args.attention_type,
            norm_type=args.norm_type,
            positional_encoding=args.positional_encoding,
            num_kv_heads=args.gqa_kv_heads,
        )

    return model.to(device)


def decode_predictions(
    model: nn.Module,
    dataloader,
    data_info: dict,
    device: torch.device,
    is_blt: bool = False,
    max_samples: int = None,
) -> tuple[list[str], list[str]]:
    """Run greedy decoding and return (predictions, targets) as strings."""
    model.eval()
    predictions = []
    targets = []
    n = 0

    tokenizer_tgt = data_info.get("tokenizer_tgt")

    for batch in dataloader:
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        if is_blt:
            pred_ids = model.greedy_decode(src, max_len=args.max_seq_len)
        else:
            pred_ids = model.greedy_decode(
                src,
                bos_idx=data_info["bos_idx"],
                eos_idx=data_info["eos_idx"],
                max_len=args.max_seq_len,
            )

        # Decode to strings
        for i in range(pred_ids.size(0)):
            pred_seq = pred_ids[i].cpu().tolist()
            tgt_seq = tgt[i].cpu().tolist()

            if is_blt:
                # Remove BOS/EOS/PAD sentinels
                pred_str = bytes([b for b in pred_seq if b < 256 and b != BYTE_PAD]).decode("utf-8", errors="replace")
                tgt_str = bytes([b for b in tgt_seq if b < 256 and b != BYTE_PAD]).decode("utf-8", errors="replace")
            else:
                # Remove special tokens and decode with tokenizer
                eos_id = data_info["eos_idx"]
                pad_id = data_info["pad_idx"]
                bos_id = data_info["bos_idx"]

                # Truncate at EOS
                if eos_id in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_id)]
                pred_seq = [t for t in pred_seq if t not in (pad_id, bos_id, eos_id)]

                if eos_id in tgt_seq:
                    tgt_seq = tgt_seq[:tgt_seq.index(eos_id)]
                tgt_seq = [t for t in tgt_seq if t not in (pad_id, bos_id, eos_id)]

                pred_str = tokenizer_tgt.decode(pred_seq) if pred_seq else ""
                tgt_str = tokenizer_tgt.decode(tgt_seq) if tgt_seq else ""

            predictions.append(pred_str)
            targets.append(tgt_str)

            n += 1
            if max_samples and n >= max_samples:
                return predictions, targets

    return predictions, targets


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    grad_clip_norm: float,
    global_step: int,
    is_blt: bool = False,
    epoch: int = 1,
    blt_patch_size: int = 9,
) -> tuple[float, int, dict]:
    """Train for one epoch.

    Returns:
        (avg_loss, global_step, speed_metrics)
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_bytes_processed = 0
    epoch_start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_idx, (src, tgt) in enumerate(train_loader):
        # Length curriculum: scale up sequence length over first 5 epochs
        # epoch 1 = 20% of seq, epoch 2 = 40%, ..., epoch >= 5 = 100%
        curriculum_ratio = min(1.0, epoch / 5.0)
        tgt_len = int(tgt.size(1) * curriculum_ratio)
        tgt_len = max(2, tgt_len) # At least length 2 for teacher forcing
        
        if is_blt:
            src_len = tgt_len * blt_patch_size
        else:
            src_len = tgt_len
            
        src = src[:, :src_len].to(device)
        tgt = tgt[:, :tgt_len].to(device)

        # Teacher forcing: input is tgt[:-1], labels are tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # Forward pass
        logits = model(src, tgt_input)

        if logits.size(1) > tgt_labels.size(1):
            logits = logits[:, :tgt_labels.size(1), :]

        # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
        vocab_size = logits.size(-1)
        loss = criterion(
            logits.contiguous().view(-1, vocab_size),
            tgt_labels.contiguous().view(-1),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        # Tracking
        if is_blt:
            pad_id = BYTE_PAD
        else:
            pad_id = criterion.ignore_index

        non_pad = (tgt_labels != pad_id).sum().item()
        total_tokens += non_pad
        total_loss += loss.item() * non_pad
        global_step += 1

        # Estimate bytes processed
        total_bytes_processed += src.numel() + tgt.numel()

        # Log every 50 steps
        if (batch_idx + 1) % 50 == 0:
            log_wandb({
                "train/step_loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/global_step": global_step,
            }, step=global_step)

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    speed_metrics = {
        "wall_time_per_epoch": epoch_time,
        "tokens_per_sec": total_tokens / epoch_time if epoch_time > 0 else 0,
        "bytes_per_sec": total_bytes_processed / epoch_time if epoch_time > 0 else 0,
    }

    if torch.cuda.is_available():
        speed_metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return avg_loss, global_step, speed_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    is_blt: bool = False,
) -> float:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        logits = model(src, tgt_input)
        
        if logits.size(1) > tgt_labels.size(1):
            logits = logits[:, :tgt_labels.size(1), :]
            
        vocab_size = logits.size(-1)
        loss = criterion(
            logits.contiguous().view(-1, vocab_size),
            tgt_labels.contiguous().view(-1),
        )

        if is_blt:
            pad_id = BYTE_PAD
        else:
            pad_id = criterion.ignore_index

        non_pad = (tgt_labels != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def main():
    global args
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"{'=' * 60}")
    print(f"Config: {args.run_name}")
    print(f"  positional_encoding: {args.positional_encoding}")
    print(f"  attention_type: {args.attention_type}")
    print(f"  norm_type: {args.norm_type}")
    print(f"  tokenization: {args.tokenization}")
    print(f"  device: {device}")
    print(f"{'=' * 60}")

    if args.tokenization == "blt":
        args.max_seq_len = PLAIN_CHUNK_SIZE * args.blt_patch_size + 32  # margin for BOS/EOS + safety

    # ── Build dataset ─────────────────────────────────────────────────
    print("\nBuilding dataset...")
    data_info = build_dataloaders(
        tokenization=args.tokenization,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        vocab_size=args.bpe_vocab_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]

    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  Src vocab: {data_info['src_vocab_size']}, Tgt vocab: {data_info['tgt_vocab_size']}")

    # ── Build model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(args, data_info, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    # ── Optimizer and scheduler ───────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    is_blt = args.tokenization == "blt"

    # Loss function: cross-entropy with label smoothing, ignore padding
    pad_idx = data_info["pad_idx"]
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=args.label_smoothing,
    )

    # ── W&B init ──────────────────────────────────────────────────────
    config_dict = {
        "run_name": args.run_name,
        "positional_encoding": args.positional_encoding,
        "attention_type": args.attention_type,
        "norm_type": args.norm_type,
        "tokenization": args.tokenization,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "epochs": args.epochs,
        "num_params": num_params,
        "device": str(device),
    }
    if args.tokenization == "blt":
        config_dict["blt_patch_size"] = args.blt_patch_size
    if args.attention_type == "gqa":
        config_dict["gqa_kv_heads"] = args.gqa_kv_heads

    run = init_wandb(args.wandb_project, config_dict, name=args.run_name)
    print(f"  W&B run: {run.url}")

    # ── Checkpoint directory ──────────────────────────────────────────
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    all_train_losses = []
    all_val_losses = []
    best_speed_metrics = {}

    for epoch in range(1, args.epochs + 1):
        # Train (time only the training loop)
        train_loss, global_step, speed_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, args.grad_clip_norm, global_step, is_blt, epoch, args.blt_patch_size
        )
        all_train_losses.append(train_loss)
        
        epoch_time = speed_metrics["wall_time_per_epoch"]

        # Validate
        val_start = time.time()
        val_loss = validate(model, val_loader, criterion, device, is_blt)
        all_val_losses.append(val_loss)
        val_time = time.time() - val_start

        # Log
        log_dict = {
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "train/wall_time": epoch_time,
            "train/tokens_per_sec": speed_metrics["tokens_per_sec"],
            "train/bytes_per_sec": speed_metrics["bytes_per_sec"],
            "epoch": epoch,
        }
        if "peak_memory_mb" in speed_metrics:
            log_dict["train/peak_memory_mb"] = speed_metrics["peak_memory_mb"]
        log_wandb(log_dict, step=global_step)

        # Print progress
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_speed_metrics = speed_metrics

            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_loss,
                "config": config_dict,
            }, ckpt_path)
            print(f"    [OK] Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    [FAIL] Early stopping triggered (patience={args.patience})")
                break

    # ── Load best model ───────────────────────────────────────────────
    print("\nLoading best model for evaluation...")
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Test evaluation ───────────────────────────────────────────────
    print("Running test evaluation with greedy decoding...")
    predictions, targets = decode_predictions(
        model, test_loader, data_info, device, is_blt=is_blt,
    )

    metrics = compute_all_metrics(predictions, targets, is_token_free=is_blt)
    metrics["best_val_loss"] = best_val_loss
    metrics["best_epoch"] = ckpt["epoch"]
    metrics["num_params"] = num_params

    # Compute naive baselines
    from src.utils import compute_naive_baselines
    train_targets = data_info["splits"]["train"]["plain"]
    test_targets = data_info["splits"]["test"]["plain"]
    baselines = compute_naive_baselines(train_targets, test_targets)
    if baselines:
        metrics["baselines"] = baselines

    # Add speed metrics
    metrics["speed"] = best_speed_metrics

    print(f"\n{'=' * 60}")
    print(f"Test Results — {args.run_name}")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if k != "speed":
            print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    # Log test metrics to W&B
    test_log = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            test_log[f"test/{k}"] = v
    log_wandb(test_log)

    # Save metrics JSON
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    metrics_path = save_metrics_json(metrics, args.run_name, output_dir)
    print(f"  Metrics saved to {metrics_path}")

    # Save losses for plotting
    losses_path = os.path.join(output_dir, f"losses_{args.run_name}.json")
    with open(losses_path, "w") as f:
        json.dump({"train": all_train_losses, "val": all_val_losses}, f)

    # ── Push to HF Hub ────────────────────────────────────────────────
    print("\nPushing checkpoint to HF Hub...")
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if hf_token:
        try:
            # Determine repo ID
            if args.hf_repo_id:
                repo_id = f"{args.hf_repo_id}-{args.run_name}"
            else:
                from huggingface_hub import HfApi
                api = HfApi()
                user_info = api.whoami(token=hf_token)
                hf_user = user_info["name"]
                repo_id = f"{hf_user}/cipher-transformer-{args.run_name}"

            # Save model config alongside checkpoint
            config_path = os.path.join(ckpt_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save tokenizers if applicable
            if data_info.get("tokenizer_src"):
                data_info["tokenizer_src"].save(os.path.join(ckpt_dir, "tokenizer_src.json"))
            if data_info.get("tokenizer_tgt"):
                data_info["tokenizer_tgt"].save(os.path.join(ckpt_dir, "tokenizer_tgt.json"))

            # Upload folder
            push_folder_to_hub(ckpt_dir, repo_id, token=hf_token)
            print(f"  ✓ Pushed to https://huggingface.co/{repo_id}")

            # Log HF link to W&B
            log_wandb({"hf_repo": repo_id})
        except Exception as e:
            print(f"  ✗ HF push failed: {e}")
    else:
        print("  ⚠ HF_TOKEN not set, skipping HF Hub push")

    # ── Finish ────────────────────────────────────────────────────────
    finish_wandb()
    print(f"\n✓ Training complete for {args.run_name}")


if __name__ == "__main__":
    main()
