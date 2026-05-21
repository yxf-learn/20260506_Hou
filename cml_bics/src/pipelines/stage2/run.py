"""Stage-2 pipeline: cross-fitted nuisance + DR pseudo-outcomes + forests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def run_stage2(cfg_path: str) -> Mapping[str, object]:
    """Run stage-2 driver."""
    raise NotImplementedError
