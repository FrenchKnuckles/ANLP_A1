"""
Phase A Diagnostic Script
=========================
1. Confirm Pre-LN vs Post-LN in code (already confirmed by reading - report only)
2. Naive baseline (most-frequent-byte + unigram sampling)
3. C5 loss curve inspection
4. Sample (prediction, target) pairs for C1 and C5
5. Epoch time breakdown analysis
"""

import os
import sys
import json
import random
import statistics
from collections import Counter

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load metrics
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

output_dir = os.path.join(PROJECT_ROOT, "outputs")


# =====================================================================
# STEP 1: Pre-LN vs Post-LN (code review summary)
# =====================================================================
print("=" * 70)
print("STEP 1: Pre-LN vs Post-LN Confirmation")
print("=" * 70)
print("""
Code in src/models/__init__.py confirms Pre-LN residual pattern:

  EncoderLayer.forward():
    x_norm = self.norm1(x)
    x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, ...))
    x = x + self.dropout2(self.ffn(self.norm2(x)))

  DecoderLayer.forward():
    x_norm = self.norm1(x)
    x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, ...))
    x_norm = self.norm2(x)
    x = x + self.dropout2(self.cross_attn(x_norm, enc_output, enc_output, ...))
    x = x + self.dropout3(self.ffn(self.norm3(x)))

This is: x = x + Sublayer(Norm(x))  =>  PRE-LN  (correct)
NOT:     x = Norm(x + Sublayer(x))  =>  POST-LN (what the README says)

VERDICT: Code is correct (Pre-LN). README label 'Post-LN' is WRONG.
""")


# =====================================================================
# STEP 2: Naive baselines
# =====================================================================
print("=" * 70)
print("STEP 2: Naive Baselines")
print("=" * 70)

# Load the chunked dataset to get training target distribution
from src.dataset import chunk_dataset, split_data, DATASET_DIR, CACHE_DIR

cipher_lines, plain_lines = chunk_dataset(DATASET_DIR, CACHE_DIR)
splits = split_data(cipher_lines, plain_lines, seed=42)

train_plain = splits["train"]["plain"]
test_plain = splits["test"]["plain"]

# Build byte frequency distribution from training targets
all_train_bytes = b"".join(t.encode("utf-8") for t in train_plain)
byte_counter = Counter(all_train_bytes)
total_bytes = len(all_train_bytes)

most_common_byte = byte_counter.most_common(1)[0]
print(f"\nTraining target byte distribution:")
print(f"  Total bytes: {total_bytes}")
print(f"  Most common byte: {most_common_byte[0]} ('{chr(most_common_byte[0])}') - {most_common_byte[1]} occurrences ({100*most_common_byte[1]/total_bytes:.1f}%)")
print(f"  Top 10 bytes:")
for byte_val, count in byte_counter.most_common(10):
    char_repr = chr(byte_val) if 32 <= byte_val < 127 else f"\\x{byte_val:02x}"
    print(f"    '{char_repr}' (0x{byte_val:02x}): {count} ({100*count/total_bytes:.1f}%)")

# Helper: compute bit accuracy between two strings
def _str_to_bytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode("utf-8")

def _bytes_to_bits(b):
    bits = []
    for byte in b:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bit_level_accuracy(predictions, targets):
    total_bits = 0
    matching_bits = 0
    for pred, tgt in zip(predictions, targets):
        pred_bytes = _str_to_bytes(pred)
        tgt_bytes = _str_to_bytes(tgt)
        max_len = max(len(pred_bytes), len(tgt_bytes))
        pred_bytes = pred_bytes.ljust(max_len, b"\x00")
        tgt_bytes = tgt_bytes.ljust(max_len, b"\x00")
        pred_bits = _bytes_to_bits(pred_bytes)
        tgt_bits = _bytes_to_bits(tgt_bytes)
        total_bits += len(pred_bits)
        matching_bits += sum(p == t for p, t in zip(pred_bits, tgt_bits))
    return matching_bits / total_bits if total_bits > 0 else 0.0

def sequence_accuracy(predictions, targets):
    if not predictions:
        return 0.0
    exact = sum(1 for p, t in zip(predictions, targets) if p == t)
    return exact / len(predictions)

def levenshtein_metrics(predictions, targets):
    import Levenshtein
    total_raw = 0.0
    total_norm = 0.0
    for pred, tgt in zip(predictions, targets):
        dist = Levenshtein.distance(pred, tgt)
        total_raw += dist
        max_len = max(len(pred), len(tgt))
        total_norm += (dist / max_len) if max_len > 0 else 0.0
    n = len(predictions) if predictions else 1
    return total_raw / n, total_norm / n

# Baseline (a): Always predict the most common byte repeated to match target length
most_freq_chr = chr(most_common_byte[0])
baseline_a_preds = [most_freq_chr * len(t) for t in test_plain]

# Baseline (b): Sample from unigram byte distribution
byte_vals = list(byte_counter.keys())
byte_probs = [byte_counter[b] / total_bytes for b in byte_vals]

random.seed(42)
def sample_unigram(length):
    sampled = random.choices(byte_vals, weights=byte_probs, k=length)
    return bytes(sampled).decode("utf-8", errors="replace")

baseline_b_preds = [sample_unigram(len(t)) for t in test_plain]

# Compute metrics for both baselines
print(f"\n--- Baseline (a): Always predict most-frequent byte '{most_freq_chr}' ---")
ba_bit = bit_level_accuracy(baseline_a_preds, test_plain)
ba_seq = sequence_accuracy(baseline_a_preds, test_plain)
ba_lev_raw, ba_lev_norm = levenshtein_metrics(baseline_a_preds, test_plain)
print(f"  Bit Accuracy:      {ba_bit:.4f}")
print(f"  Sequence Accuracy: {ba_seq:.4f}")
print(f"  Levenshtein (raw): {ba_lev_raw:.2f}")
print(f"  Levenshtein (norm):{ba_lev_norm:.4f}")

print(f"\n--- Baseline (b): Unigram byte sampling ---")
bb_bit = bit_level_accuracy(baseline_b_preds, test_plain)
bb_seq = sequence_accuracy(baseline_b_preds, test_plain)
bb_lev_raw, bb_lev_norm = levenshtein_metrics(baseline_b_preds, test_plain)
print(f"  Bit Accuracy:      {bb_bit:.4f}")
print(f"  Sequence Accuracy: {bb_seq:.4f}")
print(f"  Levenshtein (raw): {bb_lev_raw:.2f}")
print(f"  Levenshtein (norm):{bb_lev_norm:.4f}")

# Compare with trained models
print(f"\n--- Comparison ---")
for name in ["c1", "c2", "c3", "c4", "c5"]:
    m = load_json(os.path.join(output_dir, f"metrics_{name}.json"))
    print(f"  {name.upper()}: bit_acc={m['bit_accuracy']:.4f}  seq_acc={m['sequence_accuracy']:.4f}  lev_norm={m['levenshtein_normalized']:.4f}")
print(f"  Baseline(a): bit_acc={ba_bit:.4f}  seq_acc={ba_seq:.4f}  lev_norm={ba_lev_norm:.4f}")
print(f"  Baseline(b): bit_acc={bb_bit:.4f}  seq_acc={bb_seq:.4f}  lev_norm={bb_lev_norm:.4f}")


# =====================================================================
# STEP 3: C5 loss curve inspection
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: C5 Loss Curve Inspection")
print("=" * 70)

for name in ["c1", "c5"]:
    losses = load_json(os.path.join(output_dir, f"losses_{name}.json"))
    train_l = losses["train"]
    val_l = losses["val"]
    print(f"\n{name.upper()} Loss Curve:")
    print(f"  Train: {train_l[0]:.3f} -> {train_l[-1]:.3f}  (reduction: {train_l[0] - train_l[-1]:.3f})")
    print(f"  Val:   {val_l[0]:.3f} -> {val_l[-1]:.3f}  (reduction: {val_l[0] - val_l[-1]:.3f})")
    print(f"  Val loss converged? Last 5 val losses: {[f'{v:.3f}' for v in val_l[-5:]]}")
    if len(val_l) > 5:
        recent_drop = val_l[-5] - val_l[-1]
        print(f"  Last 5-epoch drop: {recent_drop:.4f}")

print("\nC5 val loss is DECREASING (4.01 -> 1.49). Not diverged or flat.")
print("C5's loss is actually MUCH LOWER than C1 (1.49 vs 3.72), but the")
print("downstream metrics (bit_acc, lev) are much worse. This suggests a")
print("problem in greedy_decode or the loss is on a different scale (byte")
print("vocab 259 vs BPE vocab 8000).")


# =====================================================================
# STEP 4: Sample predictions
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: Sample (Prediction, Target) Pairs")
print("=" * 70)

import torch
from src.dataset import build_dataloaders
from src.train import build_model

# Load C1
print("\n--- C1 Sample Predictions (loading model...) ---")
try:
    data_info_c1 = build_dataloaders(
        tokenization="subword", batch_size=8, max_seq_len=150,
        vocab_size=8000, seed=42, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build a model with same config as C1
    class Args:
        d_model = 256
        num_heads = 8
        num_encoder_layers = 4
        num_decoder_layers = 4
        d_ff = 1024
        dropout = 0.1
        max_seq_len = 150
        attention_type = "mha"
        norm_type = "layernorm"
        positional_encoding = "sinusoidal"
        gqa_kv_heads = 2
        blt_patch_size = 8
        tokenization = "subword"
    
    args_c1 = Args()
    model_c1 = build_model(args_c1, data_info_c1, device)
    
    ckpt_c1 = torch.load(
        os.path.join(PROJECT_ROOT, "checkpoints", "c1", "best_model.pt"),
        map_location=device, weights_only=False
    )
    model_c1.load_state_dict(ckpt_c1["model_state_dict"])
    model_c1.eval()
    
    test_loader = data_info_c1["test_loader"]
    tokenizer_tgt = data_info_c1["tokenizer_tgt"]
    
    # Get a single batch
    src_batch, tgt_batch = next(iter(test_loader))
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)
    
    with torch.no_grad():
        pred_ids = model_c1.greedy_decode(
            src_batch,
            bos_idx=data_info_c1["bos_idx"],
            eos_idx=data_info_c1["eos_idx"],
            max_len=150,
        )
    
    eos_id = data_info_c1["eos_idx"]
    pad_id = data_info_c1["pad_idx"]
    bos_id = data_info_c1["bos_idx"]
    
    for i in range(min(5, pred_ids.size(0))):
        pred_seq = pred_ids[i].cpu().tolist()
        tgt_seq = tgt_batch[i].cpu().tolist()
        
        if eos_id in pred_seq:
            pred_seq = pred_seq[:pred_seq.index(eos_id)]
        pred_seq = [t for t in pred_seq if t not in (pad_id, bos_id, eos_id)]
        
        if eos_id in tgt_seq:
            tgt_seq = tgt_seq[:tgt_seq.index(eos_id)]
        tgt_seq = [t for t in tgt_seq if t not in (pad_id, bos_id, eos_id)]
        
        pred_str = tokenizer_tgt.decode(pred_seq) if pred_seq else ""
        tgt_str = tokenizer_tgt.decode(tgt_seq) if tgt_seq else ""
        
        print(f"\n  Sample {i+1}:")
        print(f"    TARGET: {tgt_str[:120]}")
        print(f"    PRED:   {pred_str[:120]}")
    
    del model_c1
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  Error loading C1: {e}")


# Load C5
print("\n--- C5 Sample Predictions (loading model...) ---")
try:
    data_info_c5 = build_dataloaders(
        tokenization="blt", batch_size=8, max_seq_len=1024,
        vocab_size=8000, seed=42, num_workers=0,
    )
    
    args_c5 = Args()
    args_c5.tokenization = "blt"
    args_c5.max_seq_len = 1024
    model_c5 = build_model(args_c5, data_info_c5, device)
    
    ckpt_c5 = torch.load(
        os.path.join(PROJECT_ROOT, "checkpoints", "c5", "best_model.pt"),
        map_location=device, weights_only=False
    )
    model_c5.load_state_dict(ckpt_c5["model_state_dict"])
    model_c5.eval()
    
    test_loader_c5 = data_info_c5["test_loader"]
    src_batch, tgt_batch = next(iter(test_loader_c5))
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)
    
    from src.dataset import BYTE_PAD, BYTE_BOS, BYTE_EOS
    
    with torch.no_grad():
        pred_ids = model_c5.greedy_decode(src_batch, max_len=150)
    
    for i in range(min(5, pred_ids.size(0))):
        pred_seq = pred_ids[i].cpu().tolist()
        tgt_seq = tgt_batch[i].cpu().tolist()
        
        pred_str = bytes([b for b in pred_seq if b < 256 and b != BYTE_PAD]).decode("utf-8", errors="replace")
        tgt_str = bytes([b for b in tgt_seq if b < 256 and b != BYTE_PAD]).decode("utf-8", errors="replace")
        
        # Also show raw byte values for first 20 predicted bytes
        raw_pred = [b for b in pred_seq[:30]]
        
        print(f"\n  Sample {i+1}:")
        print(f"    TARGET: {tgt_str[:120]}")
        print(f"    PRED:   {pred_str[:120]}")
        print(f"    RAW PRED BYTES (first 30): {raw_pred}")
    
    del model_c5
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  Error loading C5: {e}")
    import traceback
    traceback.print_exc()


# =====================================================================
# STEP 5: Epoch time analysis
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Epoch Time Analysis")
print("=" * 70)

print("\nReported speed metrics from training:")
for name in ["c1", "c2", "c3", "c4", "c5"]:
    m = load_json(os.path.join(output_dir, f"metrics_{name}.json"))
    speed = m.get("speed", {})
    print(f"  {name.upper()}: wall_time={speed.get('wall_time_per_epoch', 0):.1f}s  "
          f"tokens/s={speed.get('tokens_per_sec', 0):.0f}  "
          f"bytes/s={speed.get('bytes_per_sec', 0):.0f}  "
          f"peak_mem={speed.get('peak_memory_mb', 0):.0f}MB")

print("""
ANALYSIS:
  C5 wall_time_per_epoch = 416s vs C1 = 61s (6.8x slower).
  BUT C5 bytes/sec = 57,394 vs C1 = 41,137 (C5 is 1.4x FASTER by throughput).

  The wall_time_per_epoch in train.py (line 520) measures:
    epoch_start = time.time()  (BEFORE train_one_epoch)
    ...train...
    ...validate...
    epoch_time = time.time() - epoch_start  (AFTER validate)

  So epoch_time INCLUDES the validation pass. But tokens_per_sec and
  bytes_per_sec are measured ONLY inside train_one_epoch (lines 291-355).

  For C5, the greedy decode validation is byte-by-byte autoregressive and
  very slow. The training throughput is fine, but the validation decode
  inflates the epoch wall time.

  FIX: Separate "training time" from "eval time" in the timing metrics,
  or at least note this in the README.
""")

print("=" * 70)
print("PHASE A DIAGNOSTICS COMPLETE")
print("=" * 70)
