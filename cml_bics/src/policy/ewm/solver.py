"""Empirical welfare maximisers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def solve_linear_halfspace(X, score, B_frac=0.5) -> object:
    """Linear half-space EWM."""
    raise NotImplementedError

def solve_decision_tree(X, score, depth=4, B_frac=0.5) -> object:
    """Tree-class EWM."""
    raise NotImplementedError

def greedy_under_budget(score, cost, B) -> np.ndarray:
    """Greedy budgeted assignment."""
    raise NotImplementedError
