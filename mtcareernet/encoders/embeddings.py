from __future__ import annotations
import torch
import torch.nn as nn


class CategoricalEmbedder(nn.Module):
    def __init__(self, cardinalities: list[int], dim: int = 32, padding_idx: int | None = None):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(c, dim, padding_idx=padding_idx) for c in cardinalities
        ])
        self.dim = dim

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # [B, F_cat]
        return torch.stack([emb(idx[:, i]) for i, emb in enumerate(self.embs)], dim=1)


class ContinuousNormalizer(nn.Module):
    """Per-feature affine + LN. Statistics are loaded from data manifest."""

    def __init__(self, n_features: int, dim: int = 32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))
        self.shift = nn.Parameter(torch.zeros(n_features))
        self.proj = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, F_num]
        z = (x - self.shift) * self.scale
        return self.proj(z.unsqueeze(-1))
