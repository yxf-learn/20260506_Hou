"""Heterogeneous relational graph encoder with typed attention."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import Module
from ..core.registry import register


class _TypedAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.a = nn.Parameter(torch.empty(2 * dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, hv: torch.Tensor, hu: torch.Tensor) -> torch.Tensor:
        z = torch.cat([hv, hu], dim=-1)
        return F.leaky_relu(z @ self.a, negative_slope=0.2)


class _RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_relations: int):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)])
        self.W0 = nn.Linear(in_dim, out_dim, bias=True)
        self.attn = nn.ModuleList([_TypedAttn(out_dim) for _ in range(n_relations)])

    def forward(self, h: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        out = self.W0(h)
        for r, A in enumerate(adjs):
            Whu = self.W[r](h)
            # softmax over neighbours within relation r
            logits = self.attn[r](Whu.unsqueeze(0).expand_as(Whu).contiguous(), Whu)
            alpha = torch.softmax(logits.masked_fill(A == 0, float("-inf")), dim=-1)
            out = out + alpha @ Whu
        return F.gelu(out)


@register("encoder", "rgcn_attn")
class RelationalGraphEncoder(Module):
    def __init__(self, in_dim: int = 64, hidden: int = 64, out_dim: int = 64,
                 n_relations: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            _RGCNLayer(dims[i], dims[i + 1], n_relations) for i in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self._d = out_dim

    @property
    def out_dim(self) -> int:
        return self._d

    def forward(self, h: torch.Tensor, adjs: list[torch.Tensor],
                student_idx: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = self.drop(layer(h, adjs))
        return h.index_select(0, student_idx)
