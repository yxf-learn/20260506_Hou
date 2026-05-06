"""Intersectional equalised-odds penalty (Sec. 3.2, Eq. (8))."""
from __future__ import annotations
from itertools import combinations
import torch
import torch.nn as nn


def ieo_violation(prob: torch.Tensor, y: torch.Tensor, g: torch.Tensor,
                  smooth_eps: float = 1e-3) -> torch.Tensor:
    """Maximum across groups of |P(hat Y=1 | Y=y, A=g) - P(hat Y=1 | Y=y, A=g')|."""
    groups = torch.unique(g)
    if len(groups) < 2:
        return torch.zeros((), device=prob.device)
    deltas = []
    for y_val in (0, 1):
        rates = []
        for gv in groups:
            sel = (y == y_val) & (g == gv)
            if sel.sum() < 2:
                continue
            rates.append(prob[sel].mean())
        if len(rates) < 2:
            continue
        rs = torch.stack(rates)
        deltas.append((rs.max() - rs.min()).abs())
    if not deltas:
        return torch.zeros((), device=prob.device)
    return torch.stack(deltas).max()


class IntersectionalEO(nn.Module):
    def __init__(self, beta: float = 1.0, square: bool = True):
        super().__init__()
        self.beta = beta
        self.square = square

    def forward(self, prob: torch.Tensor, y: torch.Tensor, g_idx: torch.Tensor) -> torch.Tensor:
        v = ieo_violation(prob, y, g_idx)
        return self.beta * (v.pow(2) if self.square else v)
