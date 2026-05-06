"""SalaryHead: log monthly salary regression."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..core.base import Module
from ..core.registry import register


@register("head", "salary")
class SalaryHead(Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._link = "linear"

    @property
    def link(self) -> str:
        return self._link

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z).squeeze(-1)
        if self._link == "sigmoid":
            out = torch.sigmoid(out)
        return out
