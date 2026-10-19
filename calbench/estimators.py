"""Estimators of the calibration error of scores s against labels y, computed on a finite sample.

L1 estimators target E|s - c(s)| and L2 estimators target sqrt(E (s - c(s))^2), where c(s) = P(y = 1 | s).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logit


def _bins(s: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        return np.minimum((s * n_bins).astype(int), n_bins - 1)
    order = np.argsort(s, kind="stable")  # equal mass: n_bins groups of (almost) equal size
    b = np.empty(len(s), dtype=int)
    b[order] = np.arange(len(s)) * n_bins // len(s)
    return b


def _bin_stats(s, y, n_bins, scheme):
    b = _bins(s, n_bins, scheme)
    n = np.bincount(b, minlength=n_bins).astype(float)
    keep = n > 0
    sy = np.bincount(b, y, n_bins)[keep] / n[keep]
    ss = np.bincount(b, s, n_bins)[keep] / n[keep]
    return n[keep], sy, ss


def ece(s: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "width") -> float:
    """The usual binned ECE (L1): sum over bins of weight * |mean label - mean score|."""
    n, sy, ss = _bin_stats(s, y, n_bins, scheme)
    return float((n * np.abs(sy - ss)).sum() / n.sum())


def ece_l2(s: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "mass") -> float:
    """Binned L2 calibration error: sqrt(sum of weight * (mean label - mean score)^2)."""
    n, sy, ss = _bin_stats(s, y, n_bins, scheme)
    return float(np.sqrt((n * (sy - ss) ** 2).sum() / n.sum()))


def ece_l2_debiased(s: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "mass") -> float:
    """Kumar, Liang & Ma (2019): subtract each bin's sampling variance of the mean label from the squared gap."""
    n, sy, ss = _bin_stats(s, y, n_bins, scheme)
    sq = (sy - ss) ** 2 - sy * (1 - sy) / np.maximum(n - 1, 1)
    return float(np.sqrt(max((n * sq).sum() / n.sum(), 0.0)))


def kernel_ece(s: np.ndarray, y: np.ndarray, bandwidth: float | None = None, max_points: int = 400) -> float:
    """L1 error of a Nadaraya-Watson estimate of c(s), with a Gaussian kernel on the logit scale.

    The regression is evaluated on `max_points` quantiles of the scores (each standing for an equal share of
    the rows) so that the cost stays O(n * max_points). The bandwidth defaults to Silverman's rule.
    """
    z = logit(np.clip(s, 1e-12, 1 - 1e-12))
    if bandwidth is None:
        iqr = np.subtract(*np.percentile(z, [75, 25]))
        bandwidth = 0.9 * min(z.std(), iqr / 1.34) * len(z) ** -0.2
    q = np.quantile(z, (np.arange(max_points) + 0.5) / max_points)
    num, den = np.zeros(max_points), np.zeros(max_points)
    for i in range(0, len(z), 20_000):  # in chunks, to keep the kernel matrix small
        k = np.exp(-0.5 * ((q[:, None] - z[None, i:i + 20_000]) / bandwidth) ** 2)
        num += k @ y[i:i + 20_000]
        den += k.sum(axis=1)
    c_hat = num / den
    return float(np.abs(c_hat - 1 / (1 + np.exp(-q))).mean())


L1 = {
    "ECE 15 equal-width bins": lambda s, y: ece(s, y, 15, "width"),
    "ECE 15 equal-mass bins": lambda s, y: ece(s, y, 15, "mass"),
    "ECE 100 equal-mass bins": lambda s, y: ece(s, y, 100, "mass"),
    "kernel ECE (logit)": kernel_ece,
}
L2 = {
    "L2 15 equal-mass bins": lambda s, y: ece_l2(s, y, 15, "mass"),
    "L2 debiased (Kumar 2019)": lambda s, y: ece_l2_debiased(s, y, 15, "mass"),
}
