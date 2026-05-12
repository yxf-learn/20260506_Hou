"""McCrary (2008) density test."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def mccrary_test(r: np.ndarray, c: float) -> dict:
    """Density discontinuity stat + p."""
    raise NotImplementedError
