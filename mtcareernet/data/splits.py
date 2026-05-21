"""Temporal / stratified / leave-one-institution-out splits."""
from __future__ import annotations
import numpy as np


def temporal_split(years: np.ndarray, train_years: tuple, val_years: tuple, test_years: tuple):
    return (np.isin(years, train_years), np.isin(years, val_years), np.isin(years, test_years))


def stratified_split(y: np.ndarray, frac=(0.7, 0.15, 0.15), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    masks = []
    for cls in np.unique(y):
        ix = idx[y == cls]
        rng.shuffle(ix)
        n = len(ix)
        a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
        masks.append((ix[:a], ix[a:b], ix[b:]))
    tr = np.concatenate([m[0] for m in masks])
    va = np.concatenate([m[1] for m in masks])
    te = np.concatenate([m[2] for m in masks])
    return tr, va, te


def loo_institution_split(inst_id: np.ndarray, target_inst: int):
    return inst_id != target_inst, inst_id == target_inst
