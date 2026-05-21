"""End-to-end MT-CareerNet model wrapper."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn as nn

from ..core.base import Module
from ..encoders import (TabularTransformerEncoder, RelationalGraphEncoder,
                        FusionLayer, CategoricalEmbedder, ContinuousNormalizer)
from ..heads import ReadinessHead, EmploymentHead, AlignmentHead, SalaryHead
from ..transfer.dann import DomainAdversarialBranch


@dataclass
class MTCareerNetConfig:
    n_cat: int = 8
    n_num: int = 24
    cat_card: tuple = (32, 64, 4, 6, 3, 3, 3, 8)
    d_model: int = 128
    d_graph: int = 64
    n_relations: int = 4
    use_dann: bool = True
    n_domains: int = 4


class MTCareerNet(Module):
    def __init__(self, cfg: MTCareerNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or MTCareerNetConfig()
        self.cat = CategoricalEmbedder(list(self.cfg.cat_card), dim=32)
        self.num = ContinuousNormalizer(self.cfg.n_num, dim=32)
        self.tx = TabularTransformerEncoder(d_model=self.cfg.d_model,
                                            n_tokens=self.cfg.n_cat + self.cfg.n_num)
        self.gnn = RelationalGraphEncoder(in_dim=self.cfg.d_graph, hidden=64,
                                          out_dim=self.cfg.d_graph,
                                          n_relations=self.cfg.n_relations)
        self.fusion = FusionLayer(self.cfg.d_model, self.cfg.d_graph, self.cfg.d_model)
        self.heads = nn.ModuleDict({
            "y_career_readiness":   ReadinessHead(self.cfg.d_model),
            "y_employment_6m":      EmploymentHead(self.cfg.d_model),
            "y_major_job_alignment":AlignmentHead(self.cfg.d_model),
            "y_log_starting_salary":SalaryHead(self.cfg.d_model),
        })
        if self.cfg.use_dann:
            self.dom = DomainAdversarialBranch(in_dim=self.cfg.d_model,
                                               n_domains=self.cfg.n_domains)
        else:
            self.dom = None

    def forward(self, batch: Mapping) -> Mapping:
        cat_emb = self.cat(batch["cat"])
        num_emb = self.num(batch["num"])
        tokens = torch.cat([cat_emb, num_emb, batch["text"].unsqueeze(1)], dim=1)
        z_tx = self.tx(tokens)
        h0 = torch.zeros(1, self.cfg.d_graph, device=z_tx.device)  # placeholder if no graph
        z_gr = h0.expand(z_tx.size(0), -1)
        z = self.fusion(z_tx, z_gr)
        out = {"z": z, "z_tx": z_tx, "z_gr": z_gr,
               "pred": {k: head(z) for k, head in self.heads.items()}}
        if self.dom is not None:
            out["dom_logits"] = self.dom(z, lam=batch.get("dann_lambda", 1.0))
        return out


def build_default_model() -> MTCareerNet:
    return MTCareerNet(MTCareerNetConfig())
