"""Dataset / DataLoader plumbing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
from torch.utils.data import Dataset, DataLoader

from .graph import HeteroGraph


@dataclass
class _Sample:
    cat: torch.Tensor
    num: torch.Tensor
    text_emb: torch.Tensor
    a: torch.Tensor
    g_idx: int
    y: dict[str, torch.Tensor]
    inst_id: int


class GraduateOutcomeDataset(Dataset):
    def __init__(self, frame, graph: HeteroGraph | None = None,
                 mode: str = "train", text_emb_path: str | None = None):
        self.frame = frame
        self.graph = graph
        self.mode = mode
        self._text_cache = None
        self._text_path = text_emb_path

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> _Sample:
        row = self.frame.iloc[idx]
        cat = torch.tensor(row["__cat__"], dtype=torch.long)
        num = torch.tensor(row["__num__"], dtype=torch.float32)
        text_emb = torch.tensor(row["__txt__"], dtype=torch.float32)
        a = torch.tensor(row["__a__"], dtype=torch.long)
        g_idx = int(row["__g_idx__"])
        y = {k: torch.tensor(row[k], dtype=torch.float32)
             for k in ("y_career_readiness", "y_employment_6m",
                       "y_major_job_alignment", "y_log_starting_salary")}
        return _Sample(cat, num, text_emb, a, g_idx, y, int(row["institution_id"]))


def _collate(batch: list[_Sample]) -> Mapping:
    return {
        "cat": torch.stack([b.cat for b in batch]),
        "num": torch.stack([b.num for b in batch]),
        "text": torch.stack([b.text_emb for b in batch]),
        "a": torch.stack([b.a for b in batch]),
        "g_idx": torch.tensor([b.g_idx for b in batch], dtype=torch.long),
        "y": {k: torch.stack([b.y[k] for b in batch])
              for k in ("y_career_readiness", "y_employment_6m",
                        "y_major_job_alignment", "y_log_starting_salary")},
        "inst_id": torch.tensor([b.inst_id for b in batch], dtype=torch.long),
    }


def build_dataloaders(train, val, test, batch_size: int = 256, num_workers: int = 4):
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, collate_fn=_collate),
        DataLoader(val, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=_collate),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=_collate),
    )
