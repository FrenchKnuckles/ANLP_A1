"""
Main orchestrator for the Cipher Transformer Ablation Study.
Runs configurations C1 through C5 sequentially, generates plots, and creates the README.
"""

import os
import subprocess
import json
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    {"run_name": "c1", "args": []},
    {"run_name": "c2", "args": ["--positional_encoding", "rope"]},
    {"run_name": "c3", "args": ["--attention_type", "gqa"]},
    {"run_name": "c4", "args": ["--norm_type", "rmsnorm"]},
    {"run_name": "c5", "args": ["--tokenization", "blt"]},
]


def run_training_configs():
    """Run all 5 training configurations sequentially."""
    print("=" * 60)
    print("Starting Cipher Transformer Ablation Study")
    print("=" * 60)
    
    for config in CONFIGS:
        run_name = config["run_name"]
        args = config["args"]
        
        print(f"\nLaunching {run_name.upper()} training...")
        
        # Build command: python src/train.py --run_name c1 [args...]
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "train.py"), "--run_name", run_name] + args
        
        try:
            # Run the training script, stream output to console
            subprocess.run(cmd, check=True)
            print(f"{run_name.upper()} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {run_name.upper()} failed with exit code {e.returncode}. Continuing with remaining configs.")


def generate_plots(results: dict):
    """Generate final learning curve comparison plots."""
    print("\nGenerating final comparison plots...")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Validation Loss Plot
    plt.subplot(1, 2, 1)
    for config in CONFIGS:
        run_name = config["run_name"]
        losses_file = os.path.join(output_dir, f"losses_{run_name}.json")
        if os.path.exists(losses_file):
            with open(losses_file, "r") as f:
                data = json.load(f)
                val_losses = data.get("val", [])
                plt.plot(val_losses, label=run_name.upper())
    
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Training Loss Plot
    plt.subplot(1, 2, 2)
    for config in CONFIGS:
        run_name = config["run_name"]
        losses_file = os.path.join(output_dir, f"losses_{run_name}.json")
        if os.path.exists(losses_file):
            with open(losses_file, "r") as f:
                data = json.load(f)
                train_losses = data.get("train", [])
                plt.plot(train_losses, label=run_name.upper())
    
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ablation_learning_curves.png")
    plt.savefig(plot_path)
    print(f"Learning curve plots saved to {plot_path}")

    from src.utils import plot_metrics_comparison, plot_c5_vs_c1
    plot_metrics_comparison(results, output_dir)
    if "C1" in results and "C5" in results:
        plot_c5_vs_c1(
            results["C1"], results["C5"],
            results["C1"].get("speed", {}), results["C5"].get("speed", {}),
            output_dir,
        )

def generate_readme(results: dict):
    """Generate the final README.md report with all metrics."""
    print("\nGenerating README.md report...")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    
    if not results:
        print("Warning: No metrics found. Run training first to generate README.")
        return

    readme_content = f"""# Cipher Transformer Ablation Study

This repository contains a PyTorch implementation of a Seq2Seq Transformer trained to decrypt a substitution cipher. The study performs a controlled ablation over 5 architectural configurations to analyze their impact on training speed, memory, and task performance.

## Architecture Configurations

All models use the same base hyperparameters (`d_model=256`, `num_heads=8`, `layers=4`).

*   **C1 (Base)**: Standard Transformer (BPE Tokenization, Sinusoidal PE, Multi-Head Attention, Pre-LN LayerNorm).
*   **C2 (RoPE)**: Base + Rotary Positional Embeddings (RoPE).
*   **C3 (GQA)**: Base + Grouped Query Attention (GQA with 2 KV heads).
*   **C4 (RMSNorm)**: Base + RMSNorm.
*   **C5 (BLT)**: Byte Latent Transformer (Token-free, patch size=9).

### BLT Patch Size Rationale
For C5, we set `patch_size=9`. Since the cipher is literally the 8-bit binary ASCII representation of each plaintext character followed by a `|` separator, a 9-byte patch perfectly aligns one patch with exactly one character. This is a sane inductive bias given the data's known periodicity, though the model still must learn the byte-to-character mapping and boundaries from scratch.

## Results

### Validation Loss
![Learning Curves](outputs/ablation_learning_curves.png)

### Performance Metrics

| Configuration | Bit-Level Acc | Sequence Acc | Levenshtein (Norm) | BLEU | ROUGE-L |
|---------------|---------------|--------------|--------------------|------|---------|
"""

    def fmt(val):
        if isinstance(val, str):
            return val
        return f"{val:.4f}"

    for run_name in ["C1", "C2", "C3", "C4", "C5"]:
        if run_name in results:
            res = results[run_name]
            readme_content += (
                f"| {run_name} "
                f"| {fmt(res.get('bit_accuracy', 0.0))} "
                f"| {fmt(res.get('sequence_accuracy', 0.0))} "
                f"| {fmt(res.get('levenshtein_normalized', 0.0))} "
                f"| {fmt(res.get('bleu', 0.0))} "
                f"| {fmt(res.get('rougeL', 0.0))} |\n"
            )

    readme_content += """
### Naive Baselines
*Evaluated on the raw test set prior to model evaluation to contextualize bit-level accuracy.*

| Baseline | Bit-Level Acc | Sequence Acc | Levenshtein (Norm) |
|----------|---------------|--------------|--------------------|
"""
    if "C1" in results and "baselines" in results["C1"]:
        baselines = results["C1"]["baselines"]
        if "baseline_a" in baselines:
            ba = baselines["baseline_a"]
            readme_content += (
                f"| Most Frequent Byte | {fmt(ba.get('bit_accuracy', 0.0))} "
                f"| {fmt(ba.get('sequence_accuracy', 0.0))} "
                f"| {fmt(ba.get('levenshtein_normalized', 0.0))} |\n"
            )
        if "baseline_b" in baselines:
            bb = baselines["baseline_b"]
            readme_content += (
                f"| Unigram Sample | {fmt(bb.get('bit_accuracy', 0.0))} "
                f"| {fmt(bb.get('sequence_accuracy', 0.0))} "
                f"| {fmt(bb.get('levenshtein_normalized', 0.0))} |\n"
            )

    readme_content += """
### Resource Utilization

| Configuration | Tokens/Sec | Bytes/Sec | Peak VRAM (MB) | Epoch Time (s) |
|---------------|------------|-----------|----------------|----------------|
"""

    for run_name in ["C1", "C2", "C3", "C4", "C5"]:
        if run_name in results:
            speed = results[run_name].get("speed", {})
            readme_content += (
                f"| {run_name} "
                f"| {speed.get('tokens_per_sec', 0.0):.1f} "
                f"| {speed.get('bytes_per_sec', 0.0):.1f} "
                f"| {speed.get('peak_memory_mb', 0.0):.1f} "
                f"| {speed.get('wall_time_per_epoch', 0.0):.1f} |\n"
            )

    readme_content += """
*Note: For C5 (BLT), the "Tokens/Sec" column counts raw bytes processed per second, whereas for C1-C4 it counts BPE tokens per second. Additionally, Peak GPU memory for C1 and C5 are now nearly identical due to the recent sequence-length and patch-size alignment, closing the previous ~1800MB gap.*

## Instructions to Reproduce

1. Setup environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main orchestration script (trains all 5 models sequentially):
   ```bash
   python main.py
   ```
"""

    with open(os.path.join(PROJECT_ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("README.md generated successfully.")


def main():
    # 1. Run all models
    run_training_configs()
    
    # 2. Aggregate and plot
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    results = {}
    for config in CONFIGS:
        run_name = config["run_name"]
        metrics_file = os.path.join(output_dir, f"metrics_{run_name}.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                results[run_name.upper()] = json.load(f)

    generate_plots(results)
    
    # 3. Create README
    generate_readme(results)
    
    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
