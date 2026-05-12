"""Kernel weights."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def triangular(u: np.ndarray) -> np.ndarray:
    """Triangular kernel."""
    raise NotImplementedError

def epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel."""
    raise NotImplementedError

def uniform(u: np.ndarray) -> np.ndarray:
    """Uniform kernel."""
    raise NotImplementedError
