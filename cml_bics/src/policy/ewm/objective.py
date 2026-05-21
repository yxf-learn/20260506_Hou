"""Equity-weighted welfare objectives."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def welfare_lambda(pi, tau, w_low, lam=1.0, cost=None, B=None) -> float:
    """Equity-weighted welfare."""
    raise NotImplementedError

def welfare_atkinson(pi, tau, eps=1.0) -> float:
    """Atkinson-style nonlinear welfare."""
    raise NotImplementedError
