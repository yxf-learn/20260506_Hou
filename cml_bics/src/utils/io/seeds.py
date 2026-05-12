"""Deterministic seed broker."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Mapping, Iterable, Optional, Callable, Tuple


def derive_seed(master, tag) -> int:
    """Stable seed derivation."""
    raise NotImplementedError

def seed_context(seed) -> object:
    """Context manager seeding numpy / random."""
    raise NotImplementedError
