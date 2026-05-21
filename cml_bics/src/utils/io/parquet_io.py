"""Lightweight I/O helpers around parquet."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def read_panel(path) -> pd.DataFrame:
    """Read panel parquet."""
    raise NotImplementedError

def write_table(df, path, fmt='parquet') -> None:
    """Write df to chosen format."""
    raise NotImplementedError
