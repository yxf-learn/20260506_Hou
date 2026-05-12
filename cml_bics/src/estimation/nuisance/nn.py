"""Shallow MLP nuisance fitter (optional)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def fit_mlp(X, y, depth=2, width=64) -> object:
    """Train a small MLP."""
    raise NotImplementedError

def predict_mlp(model, X) -> np.ndarray:
    """MLP predictions."""
    raise NotImplementedError
