"""Stage-1 pipeline: design diagnostics + LATE estimates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def run_stage1(cfg_path: str) -> Mapping[str, object]:
    """Run stage-1 driver."""
    raise NotImplementedError
