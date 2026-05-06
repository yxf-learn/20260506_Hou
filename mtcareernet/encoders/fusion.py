"""Fusion layers: concat-LN and gated."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..core.base import Module
from ..core.registry import register


@register("fusion", "concat_ln")
class FusionLayer(Module):
    def __init__(self, d_tx: int, d_gr: int, d_out: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_tx + d_gr, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, z_tx: torch.Tensor, z_gr: torch.Tensor) -> torch.Tensor:
        return self.ln(self.proj(torch.cat([z_tx, z_gr], dim=-1)))


@register("fusion", "gated")
class GatedFusion(Module):
    def __init__(self, d_tx: int, d_gr: int, d_out: int = 128):
        super().__init__()
        assert d_tx == d_gr == d_out, "gated fusion requires equal dims"
        self.gate = nn.Linear(d_tx + d_gr, d_out)

    def forward(self, z_tx, z_gr):
        g = torch.sigmoid(self.gate(torch.cat([z_tx, z_gr], dim=-1)))
        return g * z_tx + (1 - g) * z_gr
