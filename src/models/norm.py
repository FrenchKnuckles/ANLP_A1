"""
Custom normalization layers — implemented from scratch (no nn.LayerNorm).

LayerNorm: standard layer normalization with learned scale (gamma) and bias (beta).
RMSNorm: root-mean-square normalization with learned scale only (no mean-centering).

Both are used in a Pre-LN residual pattern: x = x + Sublayer(Norm(x)).
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization (Ba et al., 2016) — from scratch.

    Normalizes over the last dimension: y = gamma * (x - mean) / sqrt(var + eps) + beta.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) — from scratch.

    No mean-centering; normalizes by the root-mean-square of activations.
    y = gamma * x / sqrt(mean(x^2) + eps).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
