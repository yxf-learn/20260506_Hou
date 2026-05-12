"""Neyman-orthogonal moment functions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def dr_score(y, d, mu1, mu0, e) -> np.ndarray:
    """Augmented IPW pseudo-outcome."""
    raise NotImplementedError

def partial_residualised(y, d, my, md) -> Tuple[np.ndarray, np.ndarray]:
    """Partialling-out residuals."""
    raise NotImplementedError

def r_learner_target(y, d, my, md) -> np.ndarray:
    """R-learner pseudo target."""
    raise NotImplementedError
