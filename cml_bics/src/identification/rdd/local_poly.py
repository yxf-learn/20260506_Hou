"""Local polynomial estimators for fuzzy RDD."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def fit_local_quadratic(r: np.ndarray, y: np.ndarray, c: float, h: float, kernel: str = 'tri') -> dict:
    """Side-of-cutoff local quadratic fit; returns intercepts and SEs."""
    raise NotImplementedError

def late_fuzzy(rf: dict, fs: dict) -> Tuple[float, float]:
    """Ratio of reduced-form and first-stage jumps with delta-method SE."""
    raise NotImplementedError

def stacked_late(r, d, Y, c, h_grid) -> pd.DataFrame:
    """Multi-outcome LATE table across bandwidths."""
    raise NotImplementedError
