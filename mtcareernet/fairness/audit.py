"""Audit utilities for offline fairness reporting."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import torch

from .eo_penalty import equalised_odds_gap, demographic_parity_gap


@dataclass
class FairnessAudit:
    dpd: dict[str, float] = field(default_factory=dict)
    eod: dict[str, float] = field(default_factory=dict)
    dir_ratio: dict[str, float] = field(default_factory=dict)
    intersectional_dpd: float = 0.0

    def add(self, key: str, prob: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> None:
        self.dpd[key] = demographic_parity_gap(prob, a)
        self.eod[key] = equalised_odds_gap(prob, y, a)
        rs = []
        for av in torch.unique(a):
            rs.append(prob[a == av].mean().item())
        self.dir_ratio[key] = (min(rs) / max(rs)) if rs and max(rs) > 0 else 0.0

    def to_dict(self) -> Mapping:
        return {
            "dpd": self.dpd, "eod": self.eod, "dir": self.dir_ratio,
            "intersectional_dpd": self.intersectional_dpd,
        }
