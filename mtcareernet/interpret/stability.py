import torch
import torch.nn.functional as F


def attribution_stability(a: torch.Tensor, b: torch.Tensor, top_k: int = 10) -> float:
    """Cosine similarity of top-k attribution rankings between two seeds."""
    ka = a.abs().topk(top_k).indices
    kb = b.abs().topk(top_k).indices
    ua = torch.zeros_like(a); ua[ka] = a[ka]
    ub = torch.zeros_like(b); ub[kb] = b[kb]
    return F.cosine_similarity(ua.unsqueeze(0), ub.unsqueeze(0)).item()
