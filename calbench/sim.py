"""Scores whose calibration map is known exactly.

Each row has a true log-odds l ~ N(mu, sigma^2) and a label y ~ Bernoulli(sigmoid(l)). A model reports the score
s = sigmoid(h(l)) for a strictly increasing distortion h. Because s is a monotone function of l, the true
calibration map is c(s) = P(y = 1 | s) = sigmoid(h^-1(logit s)), so the calibration error of any recalibrated
score q(s) can be computed exactly by integrating over l.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

GRID = 4001


@dataclass(frozen=True)
class Distortion:
    """h(l) = a * l + b + wiggle * sin(l), with slope a_low instead of a for l < 0 when given (a kink)."""

    name: str
    a: float = 1.0
    b: float = 0.0
    wiggle: float = 0.0
    a_low: float | None = None

    def __post_init__(self):
        if self.wiggle >= min(self.a, self.a if self.a_low is None else self.a_low):
            raise ValueError("h must be strictly increasing: wiggle < slope")

    def h(self, l: np.ndarray) -> np.ndarray:
        slope = self.a if self.a_low is None else np.where(l < 0, self.a_low, self.a)
        return slope * l + self.b + self.wiggle * np.sin(l)


SCENARIOS = {
    d.name: d
    for d in [
        Distortion("calibrated"),
        Distortion("overconfident", a=1.6),
        Distortion("underconfident", a=0.6),
        Distortion("shifted", b=0.5),
        Distortion("kinked", a=1.6, a_low=0.8),
        Distortion("wiggly", wiggle=0.6),
    ]
}


@dataclass(frozen=True)
class World:
    """Population of true log-odds N(mu, sigma^2) and a score distortion."""

    mu: float
    sigma: float
    distortion: Distortion

    def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        l = rng.normal(self.mu, self.sigma, n)
        y = (rng.random(n) < expit(l)).astype(np.int8)
        return y, expit(self.distortion.h(l))

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quadrature over l: weights, scores s(l) and true probabilities c = sigmoid(l)."""
        l = np.linspace(self.mu - 8 * self.sigma, self.mu + 8 * self.sigma, GRID)
        w = norm.pdf(l, self.mu, self.sigma)
        return w / w.sum(), expit(self.distortion.h(l)), expit(l)

    def true_map(self, s: np.ndarray) -> np.ndarray:
        """c(s) = P(y = 1 | s), by inverting h on a fine grid."""
        l = np.linspace(self.mu - 12 * self.sigma, self.mu + 12 * self.sigma, 20_001)
        return expit(np.interp(logit(np.clip(s, 1e-15, 1 - 1e-15)), self.distortion.h(l), l))


def mu_for_prevalence(prevalence: float, sigma: float) -> float:
    """The mu that gives E[sigmoid(l)] = prevalence, by bisection."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        l = np.linspace(mid - 8 * sigma, mid + 8 * sigma, GRID)
        w = norm.pdf(l, mid, sigma)
        if (w * expit(l)).sum() / w.sum() < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def world(scenario: str, prevalence: float = 0.5, sigma: float = 2.0) -> World:
    return World(mu_for_prevalence(prevalence, sigma), sigma, SCENARIOS[scenario])


def true_errors(w: World, q=None) -> dict[str, float]:
    """Exact calibration error of q(s) (the raw score when q is None): L1, L2 and the excess log loss over c."""
    weights, s, c = w.grid()
    p = s if q is None else np.asarray(q(s), dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    kl = c * np.log(c / p) + (1 - c) * np.log((1 - c) / (1 - p))
    return {"l1": float((weights * np.abs(p - c)).sum()), "l2": float(np.sqrt((weights * (p - c) ** 2).sum())),
            "excess_logloss": float((weights * kl).sum())}
