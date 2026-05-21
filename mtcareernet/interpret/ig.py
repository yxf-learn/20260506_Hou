"""Integrated gradients (Sundararajan et al., 2017)."""
from __future__ import annotations
import torch


def integrated_gradients(model_fn, x: torch.Tensor, baseline: torch.Tensor,
                         steps: int = 50, target_idx: int | None = None) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device).view(-1, *([1] * x.dim()))
    interp = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    interp = interp.view(-1, *x.shape[1:])
    interp.requires_grad_(True)
    out = model_fn(interp)
    if target_idx is not None:
        out = out[:, target_idx]
    grads = torch.autograd.grad(out.sum(), interp, retain_graph=False)[0]
    grads = grads.view(steps + 1, *x.shape).mean(dim=0)
    return (x - baseline) * grads
