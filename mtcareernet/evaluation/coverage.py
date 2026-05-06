import numpy as np


def group_conditional_coverage(lo, hi, y, g):
    lo = np.asarray(lo); hi = np.asarray(hi); y = np.asarray(y); g = np.asarray(g)
    out = {}
    for gv in np.unique(g):
        sel = g == gv
        if sel.sum() == 0:
            continue
        out[int(gv)] = float(np.mean((y[sel] >= lo[sel]) & (y[sel] <= hi[sel])))
    return out
