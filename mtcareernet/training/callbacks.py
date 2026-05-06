"""Tiny callback collection for the trainer."""
from __future__ import annotations
from dataclasses import dataclass
import time
import torch


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, model, dl_val) -> None:
        score = float("nan")  # caller injects elsewhere
        better = (score < self.best - self.min_delta) if self.mode == "min"                  else (score > self.best + self.min_delta)
        if better:
            self.best = score; self.bad = 0
        else:
            self.bad += 1
            self.should_stop = self.bad >= self.patience


class GradientClipper:
    def __init__(self, max_norm: float = 5.0): self.max_norm = max_norm
    def __call__(self, model): torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


@dataclass
class MetricLogger:
    every: int = 50
    started_at: float = 0.0

    def on_epoch_end(self, epoch, model, dl_val):
        if not self.started_at:
            self.started_at = time.time()


class CheckpointSaver:
    def __init__(self, dirpath: str, monitor: str = "val_em_auroc", mode: str = "max"):
        self.dirpath = dirpath; self.monitor = monitor; self.mode = mode

    def on_epoch_end(self, epoch, model, dl_val):
        ...
