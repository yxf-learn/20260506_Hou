"""Standardised mean differences and density checks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def smd(x_left, x_right) -> float:
    """Standardised mean difference."""
    raise NotImplementedError

def covariate_balance(df, side_col, vars_) -> pd.DataFrame:
    """Balance table."""
    raise NotImplementedError
