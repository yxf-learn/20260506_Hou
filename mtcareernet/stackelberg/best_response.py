"""First-order student best-response approximation (Eq. (16))."""
from __future__ import annotations
from dataclasses import dataclass
import torch


def first_order_best_response(x_pre: torch.Tensor, grad_payoff: torch.Tensor,
                              hess_inv: torch.Tensor) -> torch.Tensor:
    return x_pre + grad_payoff @ hess_inv


@dataclass
class BestResponseSimulator:
    omega: torch.Tensor
    cost_curvature: float = 1.0
    step_clip: float = 0.5

    def step(self, x_pre: torch.Tensor, grad_y_x: torch.Tensor) -> torch.Tensor:
        # grad_y_x: [B, T, F]; payoff gradient = omega^T grad_y_x
        gp = (self.omega.view(1, -1, 1) * grad_y_x).sum(dim=1)
        delta = gp / self.cost_curvature
        delta = delta.clamp(min=-self.step_clip, max=self.step_clip)
        return x_pre + delta
