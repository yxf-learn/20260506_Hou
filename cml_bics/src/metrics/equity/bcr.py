"""Benefit-to-cost ratio."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def bcr(tau, pi, cost, months_pv_factor=1.0) -> float:
    """Aggregate BCR."""
    raise NotImplementedError
