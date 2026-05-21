"""Training loop. The Stackelberg outer-loop is handled in cli.stack_loop."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import torch

from ..losses import MultiTaskLoss
from ..fairness.intersectional import IntersectionalEO


@dataclass
class TrainingConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    fairness_beta: float = 1.0
    domain_lambda: float = 0.5
    amp: bool = True
    log_interval: int = 50
    extra: Mapping = field(default_factory=dict)


class Trainer:
    def __init__(self, model, cfg: TrainingConfig,
                 loss_fn: MultiTaskLoss | None = None,
                 fairness: IntersectionalEO | None = None):
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn or MultiTaskLoss()
        self.fairness = fairness or IntersectionalEO(beta=cfg.fairness_beta)

    def fit(self, dl_train, dl_val, optimizer, callbacks: list | None = None,
            scaler=None, device: str = "cpu") -> dict:
        self.model.to(device)
        history: dict = {"train": [], "val": []}
        for epoch in range(self.cfg.epochs):
            self.model.train()
            for batch in dl_train:
                self._step(batch, optimizer, scaler, device)
            with torch.no_grad():
                self.model.eval()
                for cb in (callbacks or []):
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(epoch, self.model, dl_val)
        return history

    def _step(self, batch, optimizer, scaler, device):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = self.model(batch)
        losses = self.loss_fn(out["pred"], batch["y"])
        fair = self.fairness(torch.sigmoid(out["pred"]["y_employment_6m"]),
                             batch["y"]["y_employment_6m"], batch["g_idx"])
        total = losses["total"] + fair
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()
