from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class ActionCatalogue:
    names: list[str] = field(default_factory=list)
    features: torch.Tensor | None = None  # [N, F]

    def add(self, name: str, feat: torch.Tensor) -> int:
        idx = len(self.names)
        self.names.append(name)
        self.features = feat.unsqueeze(0) if self.features is None             else torch.cat([self.features, feat.unsqueeze(0)], dim=0)
        return idx

    def redundancy(self) -> torch.Tensor:
        f = self.features
        return (f @ f.t()) / (f.norm(dim=-1, keepdim=True) @ f.norm(dim=-1, keepdim=True).t() + 1e-9)
