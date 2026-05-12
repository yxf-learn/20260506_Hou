"""Shallow policy-tree learner (VC-bounded)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def fit_policy_tree(X, score, depth=4, min_leaf=50) -> object:
    """Fit shallow policy tree."""
    raise NotImplementedError

def export_tree_text(tree) -> str:
    """Pretty-print policy tree."""
    raise NotImplementedError
