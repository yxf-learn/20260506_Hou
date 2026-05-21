"""Three Monte-Carlo data-generating processes used in Sec. 5.2.1."""
from __future__ import annotations
import numpy as np


def _latent_z(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.normal(size=(3, p)) * 1.5
    comps = rng.integers(0, 3, size=n)
    x = centers[comps] + rng.normal(size=(n, p))
    return x


def _heads_from_z(z: np.ndarray, rng: np.random.Generator, rho: float = 0.0) -> dict:
    d = z.shape[1]
    W_cr = rng.normal(size=(d,))
    W_em = rho * W_cr + (1 - rho) * rng.normal(size=(d,))
    W_mj = rho * W_cr + (1 - rho) * rng.normal(size=(d,))
    W_sal = rng.normal(size=(d,))
    cr = 1.0 / (1.0 + np.exp(-z @ W_cr - 0.1))
    em_p = 1.0 / (1.0 + np.exp(-z @ W_em + 0.05))
    mj_p = 1.0 / (1.0 + np.exp(-z @ W_mj - 0.15))
    em = (rng.random(size=z.shape[0]) < em_p).astype(np.int8)
    mj = (rng.random(size=z.shape[0]) < mj_p).astype(np.int8)
    sal = z @ W_sal + rng.normal(scale=0.3, size=z.shape[0])
    return {"cr": cr, "em": em, "mj": mj, "sal": sal}


def dgp_one(n: int = 8000, p: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = _latent_z(n, p, rng)
    a = rng.integers(0, 2, size=(n, 3))
    return {"x": x, "a": a, **_heads_from_z(x[:, :10], rng)}


def dgp_two(n: int = 8000, p: int = 40, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    x = _latent_z(n, p, rng)
    a = rng.integers(0, 2, size=(n, 3))
    return {"x": x, "a": a, **_heads_from_z(x[:, :10], rng, rho=0.7)}


def dgp_three(n: int = 8000, p: int = 40, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    x = _latent_z(n, p, rng)
    a = rng.integers(0, 2, size=(n, 3))
    out = _heads_from_z(x[:, :10], rng)
    bias = 0.4 * a[:, 0] - 0.3 * a[:, 1] - 0.5 * (a[:, 0] * a[:, 1])
    out["em"] = ((1.0 / (1.0 + np.exp(-(np.log((out["em"].astype(float) + 0.01) /
                  (1 - out["em"].astype(float) + 0.01)) + bias)))) > rng.random(n)).astype(np.int8)
    return {"x": x, "a": a, **out}
