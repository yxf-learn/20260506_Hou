import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = pred - target
        absE = e.abs()
        quad = 0.5 * e.pow(2)
        lin = self.delta * absE - 0.5 * self.delta ** 2
        loss = torch.where(absE <= self.delta, quad, lin)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
