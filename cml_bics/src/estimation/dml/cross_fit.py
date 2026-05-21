"""Cross-fitting harness."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def kfold_indices(n: int, k: int, seed: int) -> Iterable[Tuple[np.ndarray,np.ndarray]]:
    """K-fold indices."""
    raise NotImplementedError

def fit_predict_oof(model, X, y, splits) -> np.ndarray:
    """Out-of-fold predictions."""
    raise NotImplementedError

def stratified_kfold(n, k, strata, seed) -> Iterable[Tuple[np.ndarray,np.ndarray]]:
    """Stratified k-fold."""
    raise NotImplementedError
