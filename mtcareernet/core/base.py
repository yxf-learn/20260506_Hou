"""Base mixins shared across encoders, heads, and auxiliary modules."""
from __future__ import annotations
from typing import Any, Mapping
import torch
import torch.nn as nn


class Stateful:
    """Anything carrying a versioned state-dict beyond `nn.Module.state_dict`."""

    _state_version: int = 1

    def extra_state(self) -> Mapping[str, Any]:
        return {}

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        return None


class Module(nn.Module, Stateful):
    """Lightweight wrapper that pins a stable forward signature."""

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> "Module":
        for p in self.parameters():
            p.requires_grad_(False)
        return self
