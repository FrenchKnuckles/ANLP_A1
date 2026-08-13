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
| C1 | 0.6684 | 0.0000 | 0.7232 | 1.0180 | 0.1668 |
| C2 | 0.6615 | 0.0000 | 0.7265 | 1.0206 | 0.1753 |
| C3 | 0.6617 | 0.0000 | 0.7226 | 1.1717 | 0.1705 |
| C4 | 0.6617 | 0.0000 | 0.7226 | 1.1717 | 0.1705 |
| C5 | 0.5468 | 0.0000 | 0.9912 | N/A - token-free | N/A - token-free |

### Resource Utilization

| Configuration | Tokens/Sec | Bytes/Sec | Peak VRAM (MB) | Epoch Time (s) |
|---------------|------------|-----------|----------------|----------------|
| C1 | 13448.8 | 42355.5 | 5894.3 | 42.0 |
| C2 | 12674.7 | 40179.1 | 5897.8 | 44.5 |
| C3 | 13784.2 | 43599.4 | 5603.3 | 40.9 |
| C4 | 13882.5 | 43910.3 | 5603.3 | 40.6 |
| C5 | 8509.0 | 89987.1 | 5886.1 | 284.0 |

*Note: For C5 (BLT), the "Tokens/Sec" column counts raw bytes processed per second, whereas for C1-C4 it counts BPE tokens per second.*

## Instructions to Reproduce

1. Setup environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main orchestration script (trains all 5 models sequentially):
   ```bash
   python main.py
   ```
