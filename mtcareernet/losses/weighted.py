import torch
import torch.nn as nn


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits, target, sample_w=None):
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight, reduction="none"
        )
        if sample_w is not None:
            loss = loss * sample_w
        return loss.mean()


class WeightedMSE(nn.Module):
    def forward(self, pred, target, sample_w=None):
        e = (pred - target).pow(2)
        if sample_w is not None:
            e = e * sample_w
        return e.mean()
