"""Pointwise coverage of CATE intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def picp(true, lo, hi) -> float:
    """Pointwise coverage probability."""
    raise NotImplementedError

def mpiw(lo, hi) -> float:
    """Mean prediction interval width."""
    raise NotImplementedError
