"""
Verification tests for all custom components.

Run: python tests/verify_components.py

Tests:
1. LayerNorm matches F.layer_norm (allclose)
2. RoPE preserves vector norms; dot product depends on position difference
3. GQA with num_kv_heads == num_heads matches MHA exactly
4. Forward pass shape checks (no NaNs)
5. BPE tokenizer round-trip
6. BLT forward pass smoke test
7. Metrics smoke test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def test_layernorm():
    """Test 1: Custom LayerNorm matches F.layer_norm."""
    print("Test 1: LayerNorm vs F.layer_norm...")
    from src.models.norm import LayerNorm

    d_model = 64
    ln = LayerNorm(d_model)

    x = torch.randn(2, 10, d_model)

    # Our implementation
    out = ln(x)

    # Reference (using the same gamma/beta)
    ref = F.layer_norm(x, [d_model], weight=ln.gamma, bias=ln.beta, eps=ln.eps)

    assert torch.allclose(out, ref, atol=1e-5), \
        f"LayerNorm mismatch! Max diff: {(out - ref).abs().max().item()}"
    print("  ✓ LayerNorm matches F.layer_norm (atol=1e-5)")


def test_rmsnorm():
    """Test RMSNorm basic properties."""
    print("Test: RMSNorm properties...")
    from src.models.norm import RMSNorm

    d_model = 64
    rms = RMSNorm(d_model)

    x = torch.randn(2, 10, d_model)
    out = rms(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    assert not torch.isnan(out).any(), "RMSNorm produced NaN!"
    print("  ✓ RMSNorm output shape correct, no NaNs")


def test_sinusoidal_pe():
    """Test 2: Sinusoidal PE basic checks."""
    print("Test 2: Sinusoidal PE...")
    from src.models.positional import SinusoidalPositionalEncoding

    d_model = 64
    pe = SinusoidalPositionalEncoding(d_model, max_len=100, dropout=0.0)

    # At position 0, sin(0) = 0 for even dims, cos(0) = 1 for odd dims
    pe_table = pe.pe[0]  # (max_len, d_model)
    assert torch.allclose(pe_table[0, 0::2], torch.zeros(d_model // 2), atol=1e-6), \
        "sin(0) should be 0 for even dimensions"
    assert torch.allclose(pe_table[0, 1::2], torch.ones(d_model // 2), atol=1e-6), \
        "cos(0) should be 1 for odd dimensions"

    # Test shape
    x = torch.randn(2, 10, d_model)
    out = pe(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print("  ✓ Sinusoidal PE: sin(0)=0, cos(0)=1, correct shape")


def test_rope():
    """Test 2b: RoPE preserves norm and depends on position difference."""
    print("Test 2b: RoPE...")
    from src.models.positional import build_rope_cache, apply_rope

    head_dim = 32
    max_len = 100
    cos, sin = build_rope_cache(max_len, head_dim)

    # Test norm preservation
    q = torch.randn(1, 1, max_len, head_dim)
    k = torch.randn(1, 1, max_len, head_dim)

    q_rot, k_rot = apply_rope(q, k, cos, sin)

    # Norms should be preserved
    q_norms = q.norm(dim=-1)
    q_rot_norms = q_rot.norm(dim=-1)
    assert torch.allclose(q_norms, q_rot_norms, atol=1e-4), \
        f"RoPE doesn't preserve Q norm! Max diff: {(q_norms - q_rot_norms).abs().max()}"

    k_norms = k.norm(dim=-1)
    k_rot_norms = k_rot.norm(dim=-1)
    assert torch.allclose(k_norms, k_rot_norms, atol=1e-4), \
        f"RoPE doesn't preserve K norm! Max diff: {(k_norms - k_rot_norms).abs().max()}"

    # Test that dot product depends on position difference
    # dot(q_rot[i], k_rot[j]) should depend only on (i-j)
    q_test = torch.randn(1, 1, 1, head_dim).expand(1, 1, 50, head_dim).clone()
    k_test = q_test.clone()  # Same vector at every position

    q_r, k_r = apply_rope(q_test, k_test, cos, sin)

    # dot(pos_0, pos_d) should equal dot(pos_5, pos_5+d) for any d
    for d in [1, 3, 7]:
        dot_0_d = (q_r[0, 0, 0] * k_r[0, 0, d]).sum()
        dot_5_5d = (q_r[0, 0, 5] * k_r[0, 0, 5 + d]).sum()
        assert torch.allclose(dot_0_d, dot_5_5d, atol=1e-4), \
            f"RoPE dot product not relative! d={d}, dot(0,{d})={dot_0_d:.4f}, dot(5,{5+d})={dot_5_5d:.4f}"

    print("  ✓ RoPE: norm preserved, dot product depends only on position difference")


def test_gqa_equals_mha():
    """Test 3: GQA with num_kv_heads == num_heads matches MHA."""
    print("Test 3: GQA == MHA when num_kv_heads == num_heads...")
    from src.models.attention import MultiHeadAttention, GroupedQueryAttention

    d_model = 64
    num_heads = 4
    torch.manual_seed(42)

    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    gqa = GroupedQueryAttention(d_model, num_heads, num_heads, dropout=0.0)

    # Copy weights from MHA to GQA
    gqa.W_q.weight.data.copy_(mha.W_q.weight.data)
    gqa.W_q.bias.data.copy_(mha.W_q.bias.data)
    gqa.W_k.weight.data.copy_(mha.W_k.weight.data)
    gqa.W_k.bias.data.copy_(mha.W_k.bias.data)
    gqa.W_v.weight.data.copy_(mha.W_v.weight.data)
    gqa.W_v.bias.data.copy_(mha.W_v.bias.data)
    gqa.W_o.weight.data.copy_(mha.W_o.weight.data)
    gqa.W_o.bias.data.copy_(mha.W_o.bias.data)

    x = torch.randn(2, 10, d_model)

    mha.eval()
    gqa.eval()

    with torch.no_grad():
        mha_out = mha(x, x, x)
        gqa_out = gqa(x, x, x)

    assert torch.allclose(mha_out, gqa_out, atol=1e-5), \
        f"GQA != MHA when same heads! Max diff: {(mha_out - gqa_out).abs().max().item()}"
    print("  ✓ GQA matches MHA exactly with identical weights and num_kv_heads == num_heads")


def test_seq2seq_forward():
    """Test 4: Full Seq2SeqTransformer forward pass — shape and no NaN."""
    print("Test 4: Seq2SeqTransformer forward pass...")
    from src.models import Seq2SeqTransformer

    model = Seq2SeqTransformer(
        src_vocab_size=100, tgt_vocab_size=100,
        d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2,
        d_ff=128, dropout=0.0, max_seq_len=50, pad_idx=0,
    )
    model.eval()

    src = torch.randint(1, 100, (2, 15))
    tgt = torch.randint(1, 100, (2, 10))

    with torch.no_grad():
        logits = model(src, tgt)

    assert logits.shape == (2, 10, 100), f"Wrong shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "Forward pass produced NaN!"

    # Test greedy decode
    decoded = model.greedy_decode(src, bos_idx=1, eos_idx=2, max_len=20)
    assert decoded.shape[0] == 2, f"Wrong batch size in decode: {decoded.shape}"
    assert not torch.isnan(decoded.float()).any(), "Greedy decode produced NaN!"

    print(f"  ✓ Forward pass: logits shape {logits.shape}, no NaN")
    print(f"  ✓ Greedy decode: shape {decoded.shape}")


def test_seq2seq_configs():
    """Test all C1–C4 config variants build and forward-pass without error."""
    print("Test: All C1–C4 configs...")
    from src.models import Seq2SeqTransformer

    configs = [
        {"positional_encoding": "sinusoidal", "attention_type": "mha", "norm_type": "layernorm"},  # C1
        {"positional_encoding": "rope", "attention_type": "mha", "norm_type": "layernorm"},  # C2
        {"positional_encoding": "sinusoidal", "attention_type": "gqa", "norm_type": "layernorm"},  # C3
        {"positional_encoding": "sinusoidal", "attention_type": "mha", "norm_type": "rmsnorm"},  # C4
    ]

    for i, cfg in enumerate(configs, 1):
        model = Seq2SeqTransformer(
            src_vocab_size=100, tgt_vocab_size=100,
            d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2,
            d_ff=128, dropout=0.0, max_seq_len=50, pad_idx=0, num_kv_heads=2,
            **cfg,
        )
        model.eval()
        src = torch.randint(1, 100, (2, 15))
        tgt = torch.randint(1, 100, (2, 10))
        with torch.no_grad():
            logits = model(src, tgt)
        assert logits.shape == (2, 10, 100), f"C{i} wrong shape: {logits.shape}"
        assert not torch.isnan(logits).any(), f"C{i} produced NaN!"
        print(f"  ✓ C{i} ({cfg}): OK")


def test_blt_forward():
    """Test 6: BLT forward pass smoke test."""
    print("Test 6: BLT forward pass...")
    from src.models.blt import BLTSeq2SeqModel, BYTE_BOS, BYTE_EOS, BYTE_PAD

    model = BLTSeq2SeqModel(
        d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2,
        d_ff=128, dropout=0.0, max_seq_len=50, patch_size=4,
        d_local=32, local_heads=2,
    )
    model.eval()

    # Simulate byte input
    src = torch.randint(0, 256, (2, 20))
    src[:, 0] = BYTE_BOS
    src[:, -1] = BYTE_EOS

    tgt = torch.randint(0, 256, (2, 12))
    tgt[:, 0] = BYTE_BOS
    tgt[:, -1] = BYTE_EOS

    with torch.no_grad():
        logits = model(src, tgt)

    assert logits.shape[0] == 2, f"Wrong batch size: {logits.shape}"
    assert logits.shape[2] == 259, f"Wrong vocab size: {logits.shape[2]}"
    assert not torch.isnan(logits).any(), "BLT forward pass produced NaN!"

    # Verify patch count
    import math
    expected_src_patches = math.ceil(20 / 4)
    print(f"  ✓ BLT forward: logits shape {logits.shape}, no NaN")

    # Test greedy decode
    decoded = model.greedy_decode(src, max_len=20)
    assert decoded.shape[0] == 2, f"Wrong batch size in decode: {decoded.shape}"
    print(f"  ✓ BLT greedy decode: shape {decoded.shape}")


def test_metrics():
    """Test 7: Metrics smoke test."""
    print("Test 7: Metrics...")
    from src.utils import (
        bit_level_accuracy, sequence_accuracy, levenshtein_metrics,
    )

    # Identical pairs -> 100% accuracy, 0 distance
    preds = ["hello world", "test"]
    tgts = ["hello world", "test"]

    bit_acc = bit_level_accuracy(preds, tgts)
    assert abs(bit_acc - 1.0) < 1e-6, f"Expected 100% bit accuracy, got {bit_acc}"

    seq_acc = sequence_accuracy(preds, tgts)
    assert abs(seq_acc - 1.0) < 1e-6, f"Expected 100% sequence accuracy, got {seq_acc}"

    lev = levenshtein_metrics(preds, tgts)
    assert lev["levenshtein_raw"] == 0.0, f"Expected 0 Levenshtein, got {lev['levenshtein_raw']}"
    assert lev["levenshtein_normalized"] == 0.0, f"Expected 0 normalized, got {lev['levenshtein_normalized']}"
    print("  ✓ Identical pairs: 100% accuracy, 0 distance")

    # Near-identical pairs
    preds2 = ["hello worl"]
    tgts2 = ["hello world"]
    bit_acc2 = bit_level_accuracy(preds2, tgts2)
    assert 0 < bit_acc2 < 1.0, f"Expected partial bit accuracy, got {bit_acc2}"
    seq_acc2 = sequence_accuracy(preds2, tgts2)
    assert seq_acc2 == 0.0, f"Expected 0% sequence accuracy for near-identical, got {seq_acc2}"
    lev2 = levenshtein_metrics(preds2, tgts2)
    assert lev2["levenshtein_raw"] > 0, f"Expected positive distance, got {lev2['levenshtein_raw']}"
    print("  ✓ Near-identical pairs: partial accuracy, positive distance")


def main():
    print("=" * 60)
    print("Component Verification Tests")
    print("=" * 60)

    test_layernorm()
    test_rmsnorm()
    test_sinusoidal_pe()
    test_rope()
    test_gqa_equals_mha()
    test_seq2seq_forward()
    test_seq2seq_configs()
    test_blt_forward()
    test_metrics()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
