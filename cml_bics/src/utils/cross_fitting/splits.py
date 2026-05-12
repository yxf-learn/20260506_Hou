"""Repeated and clustered cross-fitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def clustered_kfold(n, k, cluster_ids, seed) -> Iterable:
    """Cluster-respecting k-fold."""
    raise NotImplementedError

def repeated_kfold(n, k, reps, seed) -> Iterable:
    """Repeated k-fold sequence."""
    raise NotImplementedError
