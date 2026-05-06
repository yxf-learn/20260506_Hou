"""Stateless-feeling preprocessor that fits and remembers per-column transforms."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Preprocessor:
    cont_cols: list[str]
    cat_cols: list[str]
    text_cols: list[str] = field(default_factory=list)
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    cat_maps_: dict[str, dict[Any, int]] = field(default_factory=dict)

    def fit(self, df) -> "Preprocessor":
        for c in self.cont_cols:
            v = df[c].astype(float)
            self.means_[c] = float(np.nanmean(v))
            self.stds_[c] = float(np.nanstd(v) + 1e-9)
        for c in self.cat_cols:
            uniq = list(sorted(df[c].dropna().unique()))
            self.cat_maps_[c] = {v: i + 1 for i, v in enumerate(uniq)}  # 0 = pad
        return self

    def transform(self, df):
        out = df.copy()
        for c in self.cont_cols:
            out[c] = (out[c].astype(float) - self.means_[c]) / self.stds_[c]
            out[c] = out[c].fillna(0.0)
        for c in self.cat_cols:
            mapping = self.cat_maps_[c]
            out[c] = out[c].map(mapping).fillna(0).astype(int)
        return out
