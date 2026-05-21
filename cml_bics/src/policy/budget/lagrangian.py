"""Budget-constrained Lagrangian solver."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def lagrangian_path(score, cost, B, grid) -> pd.DataFrame:
    """λ path over budget grid."""
    raise NotImplementedError

def dual_solve(score, cost, B, tol=1e-6) -> Tuple[np.ndarray, float]:
    """Dual bisection."""
    raise NotImplementedError
