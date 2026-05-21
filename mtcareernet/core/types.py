from __future__ import annotations
from enum import Enum
from typing import NewType
import torch

Tensor = torch.Tensor


class TaskKey(str, Enum):
    CR = "career_readiness"
    EM = "employment_6m"
    MJ = "major_job_alignment"
    SAL = "log_starting_salary"


class GroupKey(str, Enum):
    GENDER = "gender"
    FIRSTGEN = "first_generation"
    RURAL = "rural_origin"


class Phase(str, Enum):
    FIT = "fit"
    VAL = "val"
    CAL = "cal"
    TEST = "test"
    DEPLOY = "deploy"


NodeId = NewType("NodeId", int)
EdgeKey = NewType("EdgeKey", str)
