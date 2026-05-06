import torch


def h_delta_h_proxy(disc_logits_src: torch.Tensor, disc_logits_tgt: torch.Tensor) -> float:
    """Cheap proxy for d_{H Δ H} via discriminator accuracy."""
    s = (disc_logits_src.argmax(-1) == 0).float().mean().item()
    t = (disc_logits_tgt.argmax(-1) == 1).float().mean().item()
    return 2.0 * (0.5 * (s + t) - 0.5)
