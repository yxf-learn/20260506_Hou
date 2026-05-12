"""Integrated squared error of CATE."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def ise(tau_hat_grid, tau_true_grid, weights=None) -> float:
    """Weighted integrated squared error."""
    raise NotImplementedError
