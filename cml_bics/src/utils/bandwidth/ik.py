"""Imbens-Kalyanaraman style bandwidth utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def ik_bandwidth(r, y, c) -> float:
    """IK plug-in bandwidth."""
    raise NotImplementedError

def ck_double(r, y, c) -> float:
    """Calonico-Cattaneo doubled-bandwidth helper."""
    raise NotImplementedError
