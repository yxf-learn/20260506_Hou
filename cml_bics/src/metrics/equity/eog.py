"""Equity-adjusted Outcomes Gain."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def equity_adj_gain(tau, pi, gamma=0.3) -> float:
    """Mean gain minus γ·variance."""
    raise NotImplementedError
