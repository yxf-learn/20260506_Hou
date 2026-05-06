import numpy as np
from .metrics import auroc


def bootstrap_auroc_ci(y, p, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, len(y), size=len(y))
        vals[b] = auroc(y[ix], p[ix])
    lo = float(np.nanquantile(vals, alpha / 2))
    hi = float(np.nanquantile(vals, 1 - alpha / 2))
    return lo, hi
