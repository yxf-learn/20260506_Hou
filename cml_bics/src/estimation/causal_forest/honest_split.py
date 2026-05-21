"""Honest splitting rule maximising CATE variance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def split_score(left, right) -> float:
    """Variance-of-CATE score."""
    raise NotImplementedError

def subsample_tree(X, Y, ratio, seed) -> object:
    """Sub-sampled honest tree."""
    raise NotImplementedError
