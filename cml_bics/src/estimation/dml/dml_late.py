"""DML estimator at and near the cutoff."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def dml_late(df, c, h, learners) -> dict:
    """DML local IV estimator within window."""
    raise NotImplementedError

def dml_ate(df, learners) -> dict:
    """Average treatment effect via DR score."""
    raise NotImplementedError
