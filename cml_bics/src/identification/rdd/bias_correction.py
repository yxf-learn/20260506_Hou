"""Robust bias-corrected inference (CCT)"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def cct_bandwidth(r, y, c, rho=1.0) -> dict:
    """MSE-optimal main and bias bandwidths."""
    raise NotImplementedError

def robust_ci(point, se_bc, alpha=0.05) -> Tuple[float,float]:
    """Robust CI."""
    raise NotImplementedError

def rho_grid(r, y, c) -> pd.DataFrame:
    """Sensitivity grid for ρ."""
    raise NotImplementedError
