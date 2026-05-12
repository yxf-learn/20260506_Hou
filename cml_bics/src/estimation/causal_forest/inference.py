"""Infinitesimal jackknife variance for forests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def ij_variance(forest, X_target) -> np.ndarray:
    """Pointwise variance estimator."""
    raise NotImplementedError

def pointwise_ci(tau_hat, se, alpha=0.05) -> pd.DataFrame:
    """Pointwise CIs."""
    raise NotImplementedError
