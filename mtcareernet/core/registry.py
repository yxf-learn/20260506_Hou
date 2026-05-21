"""Tiny string-keyed registry for encoders, heads, losses, and ops."""
from __future__ import annotations
from typing import Callable, Dict, Type

REGISTRY: Dict[str, Dict[str, Type]] = {}


def register(kind: str, name: str) -> Callable:
    bucket = REGISTRY.setdefault(kind, {})

    def deco(cls):
        if name in bucket:
            raise KeyError(f"duplicate {kind}::{name}")
        bucket[name] = cls
        return cls

    return deco


def build(kind: str, name: str, **kw):
    if kind not in REGISTRY or name not in REGISTRY[kind]:
        raise KeyError(f"unknown {kind}::{name}")
    return REGISTRY[kind][name](**kw)
