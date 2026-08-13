"""
Positional encoding modules — Sinusoidal (additive) and Rotary (RoPE, applied inside attention).

These are mutually exclusive per config:
- Sinusoidal: added to token embeddings at the embedding stage (C1/C3/C4/C5).
- RoPE: applied inside attention on Q/K only (C2), with NO additive positional embedding.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding (Vaswani et al., 2017).

    Precomputes a buffer of shape (1, max_len, d_model) and adds it to input embeddings.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # Compute div_term: 10000^(2i/d_model) = exp(2i * -log(10000)/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter — not trained)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def build_rope_cache(
    max_len: int, head_dim: int, theta: float = 10000.0, device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for Rotary Position Embedding (RoPE).

    RoPE rotates pairs of dimensions in Q and K by position-dependent angles.
    θ_i = 1 / (theta^(2i/d)) for i in [0, d/2).

    Args:
        max_len: maximum sequence length.
        head_dim: dimension per attention head (must be even).
        theta: base frequency (default 10000).
        device: target device.

    Returns:
        (cos_cache, sin_cache): each of shape (max_len, head_dim), broadcastable
        over batch and head dimensions.
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

    # Frequency for each pair of dimensions: theta_i = 1 / (theta^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim))
    # (head_dim / 2,)

    positions = torch.arange(max_len, dtype=torch.float, device=device)  # (max_len,)
    angles = torch.outer(positions, freqs)  # (max_len, head_dim/2)

    # Duplicate for pairing: each pair (2i, 2i+1) gets the same angle
    cos_cache = torch.cos(angles).repeat(1, 2)  # (max_len, head_dim)
    sin_cache = torch.sin(angles).repeat(1, 2)  # (max_len, head_dim)

    return cos_cache, sin_cache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions: [x0, x1, x2, x3, ...] -> [-x_{d/2}, ..., x0, x1, ...]

    For RoPE, we negate and swap the two halves of the last dimension.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    The rotation is: RoPE(x, pos) = x * cos(pos) + rotate_half(x) * sin(pos)
    This ensures that dot(RoPE(q, i), RoPE(k, j)) depends only on (i - j).

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim) or broadcastable
        sin: (seq_len, head_dim) or broadcastable

    Returns:
        Rotated (q, k) with same shapes.

    Sanity checks:
        - Rotation preserves vector norm: ||RoPE(x)|| == ||x||.
        - dot(RoPE(q, i), RoPE(k, j)) depends only on i - j.
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    seq_len = q.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
