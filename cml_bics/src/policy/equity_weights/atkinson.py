"""Atkinson and rank-dependent equity adjustments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def atkinson_index(x, eps=1.0) -> float:
    """Atkinson inequality index."""
    raise NotImplementedError

def rank_weights(rank: np.ndarray, gamma: float) -> np.ndarray:
    """Rank-dependent weights."""
    raise NotImplementedError
