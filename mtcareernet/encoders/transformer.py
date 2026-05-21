"""Transformer channel over per-modality tokens (Sec. 3.1)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from ..core.base import Module
from ..core.registry import register


class _PositionalBias(nn.Module):
    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        return x + self.bias


@register("encoder", "tabular_tx")
class TabularTransformerEncoder(Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        n_tokens: int = 16,
        cls_token: bool = True,
    ):
        super().__init__()
        self.cls_token = cls_token
        self.pos = _PositionalBias(n_tokens + int(cls_token), d_model)
        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self._d = d_model

    @property
    def out_dim(self) -> int:
        return self._d

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b = tokens.size(0)
        if self.cls_token:
            cls = self.cls.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            if mask is not None:
                pad = torch.zeros(b, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([pad, mask], dim=1)
        x = self.pos(tokens)
        x = self.tx(x, src_key_padding_mask=mask)
        x = self.out_norm(x)
        return x[:, 0] if self.cls_token else x.mean(dim=1)
