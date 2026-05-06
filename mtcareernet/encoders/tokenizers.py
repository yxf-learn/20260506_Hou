"""CV tokenizer wrapper around a frozen pretrained backbone."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn


class CVTokenizer(nn.Module):
    def __init__(self, backbone_name: str = "bert-base-multilingual-cased",
                 freeze: bool = True, max_length: int = 256, project_to: int | None = None):
        super().__init__()
        self.backbone_name = backbone_name
        self.max_length = max_length
        self.freeze = freeze
        self._lazy: Any = None
        self.proj = nn.Linear(768, project_to) if project_to else nn.Identity()

    def _ensure(self):
        if self._lazy is None:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            self._lazy = (
                AutoTokenizer.from_pretrained(self.backbone_name),
                AutoModel.from_pretrained(self.backbone_name),
            )
            if self.freeze:
                for p in self._lazy[1].parameters():
                    p.requires_grad_(False)

    def forward(self, text_batch: list[str]) -> torch.Tensor:
        self._ensure()
        tok, model = self._lazy
        enc = tok(text_batch, padding=True, truncation=True,
                  max_length=self.max_length, return_tensors="pt")
        with torch.set_grad_enabled(not self.freeze):
            out = model(**enc).last_hidden_state[:, 0]
        return self.proj(out)
