"""Sensitivity analysis driver."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def run_sensitivity(cfg_path: str, kind: str) -> pd.DataFrame:
    """Run a sensitivity sweep."""
    raise NotImplementedError
