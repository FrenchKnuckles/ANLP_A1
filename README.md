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
| C1 | 0.9482 | 0.5505 | 0.0101 | 92.9799 | 0.9700 |
| C2 | 0.9761 | 0.7625 | 0.0043 | 96.9508 | 0.9873 |
| C3 | 0.9472 | 0.5246 | 0.0111 | 92.2777 | 0.9678 |
| C4 | 0.9470 | 0.5413 | 0.0103 | 92.6991 | 0.9697 |
| C5 | 0.6767 | 0.0000 | 0.7535 | N/A - token-free | N/A - token-free |

### Naive Baselines
*Evaluated on the raw test set prior to model evaluation to contextualize bit-level accuracy.*

| Baseline | Bit-Level Acc | Sequence Acc | Levenshtein (Norm) |
|----------|---------------|--------------|--------------------|
| Most Frequent Byte | 0.6614 | 0.0000 | 0.8317 |
| Unigram Sample | 0.6740 | 0.0000 | 0.8362 |

### Resource Utilization

| Configuration | Tokens/Sec | Bytes/Sec | Peak VRAM (MB) | Epoch Time (s) |
|---------------|------------|-----------|----------------|----------------|
| C1 | 4376.9 | 46461.5 | 4975.0 | 128.6 |
| C2 | 4274.8 | 45401.9 | 4977.3 | 131.6 |
| C3 | 4771.6 | 50684.2 | 4964.8 | 117.9 |
| C4 | 4509.8 | 47872.5 | 4748.8 | 124.8 |
| C5 | 15457.3 | 163469.1 | 4232.7 | 156.4 |

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
