"""Cutoff-anchored CATE reconciliation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def modal_covariate(df, r_col, c, w_cols) -> np.ndarray:
    """Modal w at the cutoff."""
    raise NotImplementedError

def anchor_cate(tau_w, tau_w_anchor, tau_late) -> np.ndarray:
    """Apply anchoring offset."""
    raise NotImplementedError
