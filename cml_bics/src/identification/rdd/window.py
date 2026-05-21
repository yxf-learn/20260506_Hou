"""Window construction utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def clip_to_window(df: pd.DataFrame, c: float, h: float) -> pd.DataFrame:
    """Window clip."""
    raise NotImplementedError

def effective_n(df, h, c) -> int:
    """Effective sample size inside window."""
    raise NotImplementedError
