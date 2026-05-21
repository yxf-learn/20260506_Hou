"""Gaussian-kernel smoother used to stabilise small-cell rate estimates."""
import torch


def gaussian_kernel_smoother(values: torch.Tensor, weights: torch.Tensor,
                             bandwidth: float = 0.05) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros((), device=values.device)
    eps = 1e-9
    w = weights / (weights.sum() + eps)
    grid = torch.linspace(0, 1, 51, device=values.device)
    diffs = (values.unsqueeze(0) - grid.unsqueeze(1)) / bandwidth
    K = torch.exp(-0.5 * diffs.pow(2))
    return (w * K.sum(dim=0)).sum() / (K.sum() + eps)
