"""Block-bootstrap and wild-bootstrap helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def block_resample(idx, block_len, seed) -> np.ndarray:
    """Block bootstrap indices."""
    raise NotImplementedError

def wild_bootstrap_resid(eps, seed) -> np.ndarray:
    """Mammen wild bootstrap perturbation."""
    raise NotImplementedError
