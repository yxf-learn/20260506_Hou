"""Channel-wise decomposition: transformer / graph / head contributions."""
from __future__ import annotations
import torch


def decompose_channels(z_tx: torch.Tensor, z_gr: torch.Tensor,
                       grad_z: torch.Tensor) -> dict:
    g_tx, g_gr = grad_z.chunk(2, dim=-1)
    return {
        "transformer": (z_tx * g_tx).sum(dim=-1).detach(),
        "graph": (z_gr * g_gr).sum(dim=-1).detach(),
    }
