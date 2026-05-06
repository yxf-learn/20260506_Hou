from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class FollowerPopulation:
    omega: torch.Tensor
    mu_c: float = 1.0
    L_u: float = 1.0

    @property
    def lipschitz(self) -> float:
        return self.L_u / self.mu_c

    def best_response(self, x_pre: torch.Tensor, grad_y_x: torch.Tensor) -> torch.Tensor:
        gp = (self.omega.view(1, -1, 1) * grad_y_x).sum(dim=1)
        return x_pre + gp / self.mu_c
