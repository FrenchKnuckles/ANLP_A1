# Cipher Transformer Ablation Study

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
| C1 | 0.6620 | 0.0000 | 0.7219 | 1.2620 | 0.1701 |
| C2 | 0.6642 | 0.0000 | 0.7330 | 1.0047 | 0.1716 |
| C3 | 0.6577 | 0.0000 | 0.7342 | 0.8887 | 0.1586 |
| C4 | 0.6620 | 0.0000 | 0.7301 | 1.1259 | 0.1657 |
| C5 | 0.6855 | 0.0000 | 0.7519 | N/A - token-free | N/A - token-free |

### Resource Utilization

| Configuration | Tokens/Sec | Bytes/Sec | Peak VRAM (MB) | Epoch Time (s) |
|---------------|------------|-----------|----------------|----------------|
| C1 | 13629.7 | 43143.0 | 5317.4 | 41.3 |
| C2 | 12665.9 | 40164.1 | 5316.0 | 44.4 |
| C3 | 14069.8 | 44551.9 | 5304.0 | 40.0 |
| C4 | 13835.0 | 43792.6 | 5049.4 | 40.7 |
| C5 | 9005.3 | 95236.0 | 5885.8 | 268.4 |

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
