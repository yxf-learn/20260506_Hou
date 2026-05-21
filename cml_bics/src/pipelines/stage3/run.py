"""Stage-3 pipeline: anchoring + policy learning + audit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def run_stage3(cfg_path: str) -> Mapping[str, object]:
    """Run stage-3 driver."""
    raise NotImplementedError
