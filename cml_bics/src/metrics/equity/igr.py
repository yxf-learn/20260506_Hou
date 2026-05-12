"""Inclusion Gain Ratio."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def inclusion_gain_ratio(tau, pi, w_low) -> float:
    """Share of gain accruing to disadvantaged group."""
    raise NotImplementedError
