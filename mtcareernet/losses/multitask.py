"""Composite task loss with optional uncertainty weighting (Kendall et al.)."""
from __future__ import annotations
from typing import Mapping
import torch
import torch.nn as nn

from ..core.types import TaskKey
from .huber import HuberLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights: Mapping[str, float] | None = None,
                 huber_delta: float = 1.0):
        super().__init__()
        self.task_weights = task_weights or {
            TaskKey.CR.value: 1.0,
            TaskKey.EM.value: 1.0,
            TaskKey.MJ.value: 1.0,
            TaskKey.SAL.value: 1.0,
        }
        self._cr = nn.MSELoss()
        self._em = nn.BCEWithLogitsLoss()
        self._mj = nn.BCEWithLogitsLoss()
        self._sal = HuberLoss(delta=huber_delta)

    def forward(self, pred: dict, target: dict) -> dict:
        out = {}
        out[TaskKey.CR.value] = self._cr(pred[TaskKey.CR.value], target[TaskKey.CR.value])
        out[TaskKey.EM.value] = self._em(pred[TaskKey.EM.value], target[TaskKey.EM.value])
        out[TaskKey.MJ.value] = self._mj(pred[TaskKey.MJ.value], target[TaskKey.MJ.value])
        out[TaskKey.SAL.value] = self._sal(pred[TaskKey.SAL.value], target[TaskKey.SAL.value])
        total = sum(self.task_weights[k] * v for k, v in out.items())
        out["total"] = total
        return out


class UncertaintyWeighting(nn.Module):
    """Learnable log-variance weighting (Kendall, Gal, Cipolla 2018)."""

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        precision = torch.exp(-2 * self.log_sigma)
        return sum(0.5 * p * l for p, l in zip(precision, losses)) + self.log_sigma.sum()
