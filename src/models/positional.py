import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def build_rope_cache(max_len, head_dim, theta=10000.0, device=None):
    assert head_dim % 2 == 0

    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
    positions = torch.arange(max_len, dtype=torch.float, device=device)
    angles = torch.outer(positions, freqs)

    cos_cache = torch.cos(angles).repeat(1, 2)
    sin_cache = torch.sin(angles).repeat(1, 2)
    return cos_cache, sin_cache


def _rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    seq_len = q.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
