"""Penalised linear models for nuisances."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def fit_lasso(X, y, alpha=None) -> object:
    """Cross-validated lasso."""
    raise NotImplementedError

def fit_logit_l1(X, y) -> object:
    """L1 logistic regression."""
    raise NotImplementedError
