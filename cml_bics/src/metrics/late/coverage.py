"""LATE coverage and width metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def empirical_coverage(true, lo, hi) -> float:
    """Empirical 1-α coverage."""
    raise NotImplementedError

def interval_width(lo, hi) -> float:
    """Mean interval width."""
    raise NotImplementedError
