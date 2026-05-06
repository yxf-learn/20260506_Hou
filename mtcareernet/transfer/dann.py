"""Domain-adversarial branch (Sec. 3.7, Eq. (18))."""
from __future__ import annotations
import torch
import torch.nn as nn

from ..core.base import Module
from .grl import grad_reverse


class DomainAdversarialBranch(Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64, n_domains: int = 2):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.disc(grad_reverse(z, lam))
