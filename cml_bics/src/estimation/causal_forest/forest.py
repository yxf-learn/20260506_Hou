"""Causal forest fit on doubly-robust scores."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def fit_cf(X, Yhat_dr, weights=None, n_trees=2000, min_node=12, honesty=True) -> object:
    """Fit honest forest."""
    raise NotImplementedError

def predict_cate(forest, X) -> np.ndarray:
    """CATE prediction at X."""
    raise NotImplementedError

def forest_weights(forest, X_target) -> np.ndarray:
    """Adaptive weights α_i(w)."""
    raise NotImplementedError
