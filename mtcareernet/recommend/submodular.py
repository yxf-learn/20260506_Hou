"""Submodular top-k recommender (Sec. 3.4, Eqs. (10)-(11))."""
from __future__ import annotations
from dataclasses import dataclass
import torch


def set_utility(util: torch.Tensor, redundancy: torch.Tensor, lam: float = 0.5) -> float:
    if util.numel() == 0:
        return 0.0
    base = util.sum().item()
    if redundancy.numel() == 0:
        return base
    pen = (lam * redundancy.amax(dim=-1).clamp(min=0)).sum().item()
    return base * float(torch.exp(torch.tensor(-pen)))


def is_submodular_check(F_S: dict[frozenset, float]) -> bool:
    items = sorted(F_S.keys(), key=len)
    for s in items:
        for t in items:
            if not s.issubset(t):
                continue
            for x in F_S:
                if len(x) != 1:
                    continue
                ax = next(iter(x))
                if ax in s:
                    continue
                lhs = F_S.get(s | x, F_S[s]) - F_S[s]
                rhs = F_S.get(t | x, F_S[t]) - F_S[t]
                if lhs + 1e-9 < rhs:
                    return False
    return True


@dataclass
class GreedyTopK:
    k: int = 5
    diversity_lambda: float = 0.5

    def __call__(self, util: torch.Tensor, redundancy: torch.Tensor) -> list[int]:
        n = util.size(-1)
        chosen: list[int] = []
        avail = set(range(n))
        while len(chosen) < self.k and avail:
            best, best_g = -1, -float("inf")
            for a in avail:
                pen = (self.diversity_lambda *
                       max((redundancy[a, c].item() for c in chosen), default=0.0))
                g = util[a].item() - pen
                if g > best_g:
                    best_g, best = g, a
            chosen.append(best)
            avail.discard(best)
        return chosen
