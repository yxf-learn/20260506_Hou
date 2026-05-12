"""Placebo cutoffs and predetermined-covariate jumps."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def placebo_cutoff_jumps(df, candidate_cs, y_cols) -> pd.DataFrame:
    """Placebo jump table."""
    raise NotImplementedError
