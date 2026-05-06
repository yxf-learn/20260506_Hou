import math
import numpy as np


def calibrate_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    qi = math.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(qi, 1.0), method="higher"))


def mondrian_quantile(scores: np.ndarray, groups: np.ndarray, alpha: float,
                      min_n: int = 30) -> dict:
    out = {}
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() < min_n:
            continue
        out[int(g)] = calibrate_quantile(scores[sel], alpha)
    return out
