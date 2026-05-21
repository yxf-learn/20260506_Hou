import numpy as np


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p)
    y = y[order]
    pos = (y == 1).sum(); neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    cum_neg = np.cumsum(y == 0)
    return float((((y == 1) * cum_neg).sum()) / (pos * neg))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def mape(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(np.abs((y - p) / (np.abs(y) + 1e-9))))


def picp(lo, hi, y):
    return float(np.mean((np.asarray(y) >= np.asarray(lo)) & (np.asarray(y) <= np.asarray(hi))))


def average_width(lo, hi):
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
