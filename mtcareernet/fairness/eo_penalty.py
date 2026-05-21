import torch


def equalised_odds_gap(prob: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> float:
    out = 0.0
    for y_val in (0, 1):
        rs = []
        for av in torch.unique(a):
            sel = (y == y_val) & (a == av)
            if sel.sum() < 2:
                continue
            rs.append(prob[sel].mean().item())
        if len(rs) >= 2:
            out = max(out, max(rs) - min(rs))
    return out


def demographic_parity_gap(prob: torch.Tensor, a: torch.Tensor) -> float:
    rs = []
    for av in torch.unique(a):
        rs.append(prob[a == av].mean().item())
    return max(rs) - min(rs) if len(rs) >= 2 else 0.0
