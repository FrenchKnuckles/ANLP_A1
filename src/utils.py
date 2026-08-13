"""
Utility functions — W&B/HF Hub helpers, evaluation metrics, and plotting.

Metrics (Section 8 of the assignment):
    - Bit-Level Accuracy: expand bytes to bits, pad shorter to match longer, % matching bits.
    - Sequence Accuracy: % exact end-to-end matches.
    - Levenshtein Distance: raw + length-normalized edit distance.
    - IMPORTANT: BLEU and ROUGE-1/2/L metrics are explicitly skipped for C5 (Token-Free) as requested. They are only computed for tokenized models (C1–C4).

Plotting:
    - Loss curves per config.
    - Grouped bar chart comparing all configs.
    - C5-vs-C1 dedicated comparison.

Uses standard libraries (editdistance, sacrebleu, rouge-score) for evaluation —
the "from scratch" constraint applies to the transformer, not eval utilities.
"""

import os
import json
from typing import Optional

import torch
import wandb
from huggingface_hub import HfApi, hf_hub_download


# ═══════════════════════════════════════════════════════════════════════
# W&B Helpers
# ═══════════════════════════════════════════════════════════════════════

def init_wandb(project: str, config: dict, name: str | None = None) -> wandb.sdk.wandb_run.Run:
    return wandb.init(project=project, config=config, name=name)


def log_wandb(metrics: dict, step: int | None = None) -> None:
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# HF Hub Helpers
# ═══════════════════════════════════════════════════════════════════════

def push_to_hub(
    path: str,
    repo_id: str,
    path_in_repo: str | None = None,
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    return api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo or os.path.basename(path),
        repo_id=repo_id,
        token=token,
    )


def push_folder_to_hub(
    folder_path: str,
    repo_id: str,
    token: str | None = None,
) -> str:
    """Upload an entire folder to HF Hub."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    return api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        token=token,
    )


def pull_from_hub(
    repo_id: str,
    filename: str,
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )


def save_and_push(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    torch.save(model.state_dict(), local_path)
    return push_to_hub(local_path, repo_id, filename, token)


def load_from_hub(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    device: str = "cpu",
    token: str | None = None,
) -> torch.nn.Module:
    path = pull_from_hub(repo_id, filename, local_dir, token)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def _str_to_bytes(s: str) -> bytes:
    """Convert string to bytes (UTF-8)."""
    if isinstance(s, bytes):
        return s
    return s.encode("utf-8")


def _bytes_to_bits(b: bytes) -> list[int]:
    """Expand each byte to its 8-bit binary form."""
    bits = []
    for byte in b:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bit_level_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Bit-Level Accuracy.

    Detokenize/convert both sequences to raw bytes, right-pad the shorter one
    with zero bytes to match the longer one's length, expand each byte to its
    8-bit binary form, and report % matching bits over the aligned length.

    Alignment rule: length mismatches are handled by zero-byte right-padding
    the shorter sequence, not by truncation.
    """
    total_bits = 0
    matching_bits = 0

    for pred, tgt in zip(predictions, targets):
        pred_bytes = _str_to_bytes(pred)
        tgt_bytes = _str_to_bytes(tgt)

        # Right-pad shorter with zero bytes
        max_len = max(len(pred_bytes), len(tgt_bytes))
        pred_bytes = pred_bytes.ljust(max_len, b"\x00")
        tgt_bytes = tgt_bytes.ljust(max_len, b"\x00")

        pred_bits = _bytes_to_bits(pred_bytes)
        tgt_bits = _bytes_to_bits(tgt_bytes)

        total_bits += len(pred_bits)
        matching_bits += sum(p == t for p, t in zip(pred_bits, tgt_bits))

    return matching_bits / total_bits if total_bits > 0 else 0.0


def sequence_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Sequence Accuracy: % of pairs that are an exact end-to-end match."""
    if not predictions:
        return 0.0
    exact = sum(1 for p, t in zip(predictions, targets) if p == t)
    return exact / len(predictions)


def levenshtein_metrics(
    predictions: list[str],
    targets: list[str],
    byte_level: bool = False,
) -> dict:
    """Levenshtein Distance metrics.

    Args:
        predictions: predicted strings.
        targets: target strings.
        byte_level: if True, compute on raw bytes (for C5/BLT).
                    if False, compute on characters (for C1–C4).

    Returns:
        dict with 'levenshtein_raw' (average) and 'levenshtein_normalized'
        (dist / max(len(pred), len(target))).
    """
    import Levenshtein

    total_raw = 0.0
    total_norm = 0.0

    for pred, tgt in zip(predictions, targets):
        if byte_level:
            pred_seq = _str_to_bytes(pred)
            tgt_seq = _str_to_bytes(tgt)
        else:
            pred_seq = pred
            tgt_seq = tgt

        dist = Levenshtein.distance(pred_seq, tgt_seq)
        total_raw += dist
        max_len = max(len(pred_seq), len(tgt_seq))
        total_norm += (dist / max_len) if max_len > 0 else 0.0

    n = len(predictions) if predictions else 1
    return {
        "levenshtein_raw": total_raw / n,
        "levenshtein_normalized": total_norm / n,
    }


def compute_bleu(predictions: list[str], targets: list[str]) -> float:
    """BLEU score using sacrebleu. For tokenized models (C1–C4) only."""
    import sacrebleu

    # sacrebleu expects list of hypotheses and list of list of references
    bleu = sacrebleu.corpus_bleu(predictions, [targets])
    return bleu.score


def compute_rouge(predictions: list[str], targets: list[str]) -> dict:
    """ROUGE-1/2/L using rouge-score. For tokenized models (C1–C4) only."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_scores, r2_scores, rl_scores = [], [], []
    for pred, tgt in zip(predictions, targets):
        scores = scorer.score(tgt, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(r1_scores) / len(r1_scores) if r1_scores else 0.0,
        "rouge2": sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
        "rougeL": sum(rl_scores) / len(rl_scores) if rl_scores else 0.0,
    }


def compute_naive_baselines(train_targets: list[str], test_targets: list[str]) -> dict:
    """Compute baselines on raw text/bytes before evaluating."""
    import collections
    import random
    
    if not train_targets or not test_targets:
        return {}

    # Gather character frequencies from train targets
    counter = collections.Counter()
    for tgt in train_targets:
        for token in tgt:
            counter[token] += 1
            
    if not counter:
        return {}
        
    most_freq_token = counter.most_common(1)[0][0]
    
    # Baseline A: Most frequent byte
    preds_a = []
    for tgt in test_targets:
        preds_a.append(most_freq_token * len(tgt))

    # Baseline B: Unigram sample
    population = list(counter.keys())
    weights = list(counter.values())
    
    preds_b = []
    for tgt in test_targets:
        sampled = random.choices(population, weights=weights, k=len(tgt))
        preds_b.append("".join(sampled))
            
    metrics_a = {
        "bit_accuracy": bit_level_accuracy(preds_a, test_targets),
        "sequence_accuracy": sequence_accuracy(preds_a, test_targets),
        **levenshtein_metrics(preds_a, test_targets, byte_level=False)
    }
    
    metrics_b = {
        "bit_accuracy": bit_level_accuracy(preds_b, test_targets),
        "sequence_accuracy": sequence_accuracy(preds_b, test_targets),
        **levenshtein_metrics(preds_b, test_targets, byte_level=False)
    }
    
    return {"baseline_a": metrics_a, "baseline_b": metrics_b}


def compute_all_metrics(
    predictions: list[str],
    targets: list[str],
    is_token_free: bool = False,
) -> dict:
    """Compute all evaluation metrics for a set of predictions.

    Args:
        predictions: decoded prediction strings.
        targets: ground truth strings.
        is_token_free: if True (C5/BLT), skip BLEU/ROUGE and use byte-level Levenshtein.

    Returns:
        dict of all metric values.
    """
    metrics = {
        "bit_accuracy": bit_level_accuracy(predictions, targets),
        "sequence_accuracy": sequence_accuracy(predictions, targets),
    }

    # Levenshtein
    lev = levenshtein_metrics(predictions, targets, byte_level=is_token_free)
    metrics.update(lev)

    # BLEU/ROUGE — tokenized only
    if not is_token_free:
        metrics["bleu"] = compute_bleu(predictions, targets)
        rouge = compute_rouge(predictions, targets)
        metrics.update(rouge)
    else:
        metrics["bleu"] = "N/A - token-free"
        metrics["rouge1"] = "N/A - token-free"
        metrics["rouge2"] = "N/A - token-free"
        metrics["rougeL"] = "N/A - token-free"

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_loss_curves(
    all_losses: dict[str, dict[str, list[float]]],
    output_dir: str = "outputs",
):
    """Plot training and validation loss curves for all configs.

    Args:
        all_losses: {config_name: {'train': [...], 'val': [...]}}
        output_dir: directory to save plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for config_name, losses in all_losses.items():
        epochs = range(1, len(losses["train"]) + 1)
        axes[0].plot(epochs, losses["train"], label=config_name)
        axes[1].plot(epochs, losses["val"], label=config_name)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    all_metrics: dict[str, dict],
    output_dir: str = "outputs",
):
    """Grouped bar chart comparing all five configs across shared metrics.

    Args:
        all_metrics: {config_name: {metric_name: value}}
        output_dir: directory to save plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # Only plot numeric metrics
    numeric_metrics = ["bit_accuracy", "sequence_accuracy", "levenshtein_normalized"]
    configs = list(all_metrics.keys())

    fig, axes = plt.subplots(1, len(numeric_metrics), figsize=(5 * len(numeric_metrics), 5))
    if len(numeric_metrics) == 1:
        axes = [axes]

    x = np.arange(len(configs))
    width = 0.6

    for ax, metric in zip(axes, numeric_metrics):
        values = []
        for config in configs:
            val = all_metrics[config].get(metric, 0)
            values.append(val if isinstance(val, (int, float)) else 0)

        bars = ax.bar(x, values, width, color=plt.cm.Set2(range(len(configs))))
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_c5_vs_c1(
    c1_metrics: dict,
    c5_metrics: dict,
    c1_speed: dict,
    c5_speed: dict,
    output_dir: str = "outputs",
):
    """Dedicated C5 (BLT) vs C1 (Base) comparison plot.

    Shows quality metrics, training speed, and peak GPU memory side by side.

    Args:
        c1_metrics, c5_metrics: evaluation metrics dicts.
        c1_speed, c5_speed: dicts with 'wall_time_per_epoch', 'bytes_per_sec', 'peak_memory_mb'.
        output_dir: directory to save plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Quality metrics
    quality_metrics = ["bit_accuracy", "sequence_accuracy", "levenshtein_normalized"]
    c1_vals = [c1_metrics.get(m, 0) for m in quality_metrics]
    c5_vals = [c5_metrics.get(m, 0) if isinstance(c5_metrics.get(m, 0), (int, float)) else 0
               for m in quality_metrics]

    x = np.arange(len(quality_metrics))
    width = 0.35
    axes[0].bar(x - width / 2, c1_vals, width, label="C1 (Base)", color="#4C72B0")
    axes[0].bar(x + width / 2, c5_vals, width, label="C5 (BLT)", color="#DD8452")
    axes[0].set_title("Quality Metrics")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in quality_metrics], fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Speed comparison
    speed_metrics = ["wall_time_per_epoch", "bytes_per_sec"]
    speed_labels = ["Wall Time/Epoch (s)", "Bytes/sec"]
    c1_speed_vals = [c1_speed.get(m, 0) for m in speed_metrics]
    c5_speed_vals = [c5_speed.get(m, 0) for m in speed_metrics]

    x2 = np.arange(len(speed_metrics))
    axes[1].bar(x2 - width / 2, c1_speed_vals, width, label="C1 (Base)", color="#4C72B0")
    axes[1].bar(x2 + width / 2, c5_speed_vals, width, label="C5 (BLT)", color="#DD8452")
    axes[1].set_title("Training Speed")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(speed_labels, fontsize=8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # Memory comparison
    c1_mem = c1_speed.get("peak_memory_mb", 0)
    c5_mem = c5_speed.get("peak_memory_mb", 0)
    axes[2].bar(["C1 (Base)", "C5 (BLT)"], [c1_mem, c5_mem],
                color=["#4C72B0", "#DD8452"])
    axes[2].set_title("Peak GPU Memory (MB)")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "c5_vs_c1.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics_json(metrics: dict, config_name: str, output_dir: str = "outputs"):
    """Save metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_{config_name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path
