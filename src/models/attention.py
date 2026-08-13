"""
Custom attention modules — implemented from scratch (no nn.MultiheadAttention).

- scaled_dot_product_attention: core QKV math.
- MultiHeadAttention (MHA): standard multi-head attention.
- GroupedQueryAttention (GQA): fewer KV heads shared across query heads.

Both MHA and GQA accept an optional RoPE hook so the model assembly can
toggle rotary embeddings without duplicating attention code.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import apply_rope


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V.

    Args:
        q: (batch, num_heads, seq_len_q, d_k)
        k: (batch, num_heads, seq_len_k, d_k)
        v: (batch, num_heads, seq_len_k, d_v)
        mask: additive mask — positions with large negative values are masked out.
              Shape broadcastable to (batch, num_heads, seq_len_q, seq_len_k).
        dropout: optional dropout applied to attention weights.

    Returns:
        (output, attention_weights):
            output: (batch, num_heads, seq_len_q, d_v)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, v)  # (batch, heads, seq_q, d_v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (Vaswani et al., 2017) — from scratch.

    Supports self-attention (encoder or causal decoder) and cross-attention.
    Optionally applies RoPE to Q and K when rope_cos/rope_sin are provided.

    Args:
        d_model: model dimension.
        num_heads: number of attention heads.
        dropout: dropout rate on attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  additive mask, broadcastable to (batch, heads, seq_q, seq_k)
            rope_cos, rope_sin: precomputed RoPE tables, shape (max_len, head_dim).

        Returns:
            (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Project and reshape: (batch, seq, d_model) -> (batch, heads, seq, head_dim)
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K if provided
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Attention
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # Concatenate heads: (batch, heads, seq, head_dim) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (Ainslie et al., 2023) — from scratch.

    Uses fewer KV heads than query heads: num_query_heads query heads share
    num_kv_heads key/value heads. Each KV head serves (num_query_heads // num_kv_heads)
    query heads.

    Invariant: with num_kv_heads == num_query_heads and identical weights,
    GQA output must match MHA output exactly.

    Same call signature as MultiHeadAttention — drop-in replacement.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_query_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_query_heads ({num_query_heads})"
        assert num_query_heads % num_kv_heads == 0, \
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.groups = num_query_heads // num_kv_heads  # queries per KV head

        # Query projection: full num_query_heads
        self.W_q = nn.Linear(d_model, num_query_heads * self.head_dim)
        # KV projections: reduced num_kv_heads
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same signature as MultiHeadAttention.forward."""
        batch_size = query.size(0)

        # Project queries: (batch, seq_q, num_query_heads * head_dim)
        q = self.W_q(query).view(batch_size, -1, self.num_query_heads, self.head_dim).transpose(1, 2)
        # (batch, num_query_heads, seq_q, head_dim)

        # Project keys/values: (batch, seq_k, num_kv_heads * head_dim)
        k = self.W_k(key).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # (batch, num_kv_heads, seq_k, head_dim)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            # For Q: all query heads
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Expand KV heads to match query heads by repeating
        # (batch, num_kv_heads, seq, head_dim) -> (batch, num_query_heads, seq, head_dim)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        # Standard attention from here
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # Concatenate: (batch, num_query_heads, seq_q, head_dim) -> (batch, seq_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)
