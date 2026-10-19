"""Post-hoc calibration methods. Each `fit(s, y)` returns a function q mapping raw scores to probabilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

EPS = 1e-12


def _z(s: np.ndarray) -> np.ndarray:
    return logit(np.clip(s, EPS, 1 - EPS))


def _logistic(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000).fit(X, y)  # effectively unpenalised


def temperature(s, y):
    """One scale on the logit, no shift (Guo et al. 2017)."""
    z = _z(s)

    def nll(log_t):
        p = np.clip(expit(z * np.exp(log_t)), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    t = np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x)
    return lambda s2: expit(_z(s2) * t)


def platt(s, y):
    """Logistic regression on the logit of the score (scale and shift)."""
    m = _logistic(_z(s)[:, None], y)
    return lambda s2: m.predict_proba(_z(s2)[:, None])[:, 1]


def beta(s, y):
    """Beta calibration (Kull, Silva Filho & Flach 2017): logistic regression on log s and -log(1 - s)."""
    f = lambda v: np.column_stack([np.log(np.clip(v, EPS, 1)), -np.log(np.clip(1 - v, EPS, 1))])
    m = _logistic(f(s), y)
    return lambda s2: m.predict_proba(f(s2))[:, 1]


def isotonic(s, y):
    m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
    return lambda s2: m.predict(s2)


def histogram(s, y, n_bins: int = 15):
    """Histogram binning (Zadrozny & Elkan 2001) with equal-mass bins: each score maps to its bin's label rate."""
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, s, side="right")
    rate = np.bincount(b, y, n_bins) / np.maximum(np.bincount(b, minlength=n_bins), 1)
    return lambda s2: rate[np.searchsorted(edges, s2, side="right")]


def spline(s, y, n_knots: int = 6):
    """Logistic regression on a cubic spline basis of the logit, with knots at quantiles (extrapolated linearly)."""
    t = SplineTransformer(n_knots=n_knots, degree=3, knots="quantile", extrapolation="linear").fit(_z(s)[:, None])
    m = _logistic(t.transform(_z(s)[:, None]), y)
    return lambda s2: m.predict_proba(t.transform(_z(s2)[:, None]))[:, 1]


METHODS = {"temperature": temperature, "Platt": platt, "beta": beta, "isotonic": isotonic,
           "histogram binning": histogram, "spline": spline}
