"""Gradient boosted trees for μ and e."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def fit_gbm_classifier(X, y, params=None) -> object:
    """GBM classifier."""
    raise NotImplementedError

def fit_gbm_regressor(X, y, params=None) -> object:
    """GBM regressor."""
    raise NotImplementedError
