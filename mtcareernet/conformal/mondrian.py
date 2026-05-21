"""Per-group (Mondrian) conformal sets — Theorem 6 in the paper."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch


@dataclass
class MondrianConformal:
    alpha: float = 0.10
    min_per_cell: int = 30
    quantiles_: dict[int, float] = field(default_factory=dict)
    fallback_q_: float = 0.0

    def fit(self, y_hat_cal: torch.Tensor, y_cal: torch.Tensor,
            g_idx: torch.Tensor) -> "MondrianConformal":
        scores = (y_hat_cal - y_cal).abs().cpu().numpy()
        groups = g_idx.cpu().numpy()
        for g in np.unique(groups):
            sel = groups == g
            n = sel.sum()
            if n < self.min_per_cell:
                continue
            qi = math.ceil((n + 1) * (1 - self.alpha)) / n
            self.quantiles_[int(g)] = float(
                np.quantile(scores[sel], min(qi, 1.0), method="higher"))
        nfull = len(scores)
        qi = math.ceil((nfull + 1) * (1 - self.alpha)) / nfull
        self.fallback_q_ = float(np.quantile(scores, min(qi, 1.0), method="higher"))
        return self

    def q_for(self, g: int) -> float:
        return self.quantiles_.get(int(g), self.fallback_q_)

    def interval(self, y_hat: torch.Tensor, g_idx: torch.Tensor):
        q = torch.tensor([self.q_for(int(g)) for g in g_idx.cpu().numpy()],
                         device=y_hat.device, dtype=y_hat.dtype)
        return y_hat - q, y_hat + q
