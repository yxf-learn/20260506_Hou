"""Repeated risk minimisation with proximal stabiliser (Eq. (17))."""
from __future__ import annotations
import torch
import torch.nn as nn


class ProximalUpdate:
    """θ_{τ+1} = argmin_θ L(θ; P̂_post) + (1/2η) ||θ - θ_τ||^2."""

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def apply(self, model: nn.Module, anchor_state: dict, lam: float | None = None) -> None:
        lam = lam if lam is not None else 1.0 / (2.0 * self.eta)
        for n, p in model.named_parameters():
            if n in anchor_state and p.requires_grad:
                p.data.add_(lam * (anchor_state[n].to(p.device) - p.data))


class RepeatedRiskMin:
    def __init__(self, base_optimizer, prox: ProximalUpdate, decay: float = 0.5):
        self.opt = base_optimizer
        self.prox = prox
        self.decay = decay

    def step(self, loss: torch.Tensor, model: nn.Module, anchor: dict, t: int) -> None:
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.prox.eta = self.prox.eta * self.decay if t > 0 else self.prox.eta
        self.prox.apply(model, anchor)
