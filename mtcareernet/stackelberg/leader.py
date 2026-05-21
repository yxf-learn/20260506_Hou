from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class LeaderPolicy:
    """Policy emits a top-k action set per student under utility weights ω."""

    k: int = 5
    diversity_lambda: float = 0.5

    def __call__(self, scores: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
        # greedy submodular selection
        n_actions = scores.size(-1)
        chosen: list[int] = []
        avail = list(range(n_actions))
        while len(chosen) < self.k and avail:
            gains = []
            for a in avail:
                pen = self.diversity_lambda * max(
                    (redundancy[a, c].item() for c in chosen), default=0.0
                )
                gains.append(scores[a].item() - pen)
            best = avail[int(torch.tensor(gains).argmax())]
            chosen.append(best)
            avail.remove(best)
        return torch.tensor(chosen, dtype=torch.long)
