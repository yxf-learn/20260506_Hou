"""Split-conformal head (Sec. 3.5, Eq. (12))."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch


def _abs_score(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (yhat - y).abs()


@dataclass
class SplitConformal:
    alpha: float = 0.10

    def fit(self, y_hat_cal: torch.Tensor, y_cal: torch.Tensor) -> "SplitConformal":
        s = _abs_score(y_hat_cal, y_cal).cpu().numpy()
        n = len(s)
        q_index = math.ceil((n + 1) * (1 - self.alpha)) / n
        self._q = float(np.quantile(s, min(max(q_index, 0.0), 1.0), method="higher"))
        return self

    def interval(self, y_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return y_hat - self._q, y_hat + self._q
