"""Welfare regret of learned rules."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def regret_oracle(pi_hat, tau_true, pi_star, lam=0.0) -> float:
    """Regret vs. infeasible oracle."""
    raise NotImplementedError

def regret_curve_lambda(grid, results) -> pd.DataFrame:
    """Regret vs. λ curve."""
    raise NotImplementedError
