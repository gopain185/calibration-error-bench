"""The experiments of the benchmark. Each returns a tidy table; `summarise_*` aggregates it over replicates."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .calibrators import METHODS
from .estimators import L1, L2, ece
from .sim import SCENARIOS, true_errors, world

SEED = 20261020


def estimator_bias(ns=(1_000, 10_000, 100_000), prevalences=(0.5, 0.05), scenarios=tuple(SCENARIOS),
                   reps=(300, 300, 100)) -> pd.DataFrame:
    """Each estimator on the raw scores of a fresh sample, next to the exact calibration error."""
    rows = []
    for (sc, prev), i in zip(itertools.product(scenarios, prevalences), itertools.count()):
        w = world(sc, prev)
        truth = true_errors(w)
        for n, r in zip(ns, reps):
            rng = np.random.default_rng([SEED, 1, i, n])
            for rep in range(r):
                y, s = w.sample(rng, n)
                for norm_, ests in (("l1", L1), ("l2", L2)):
                    for name, fn in ests.items():
                        rows.append((sc, prev, n, int(y.sum()), rep, norm_, name, truth[norm_], fn(s, y)))
    return pd.DataFrame(rows, columns=["scenario", "prevalence", "n", "positives", "rep", "norm", "estimator",
                                       "truth", "estimate"])


def summarise_bias(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(err=df.estimate - df.truth).groupby(["scenario", "prevalence", "n", "norm", "estimator"],
                                                      sort=False)
    return g.agg(truth=("truth", "first"), mean_estimate=("estimate", "mean"), bias=("err", "mean"),
                 rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                 positives=("positives", "mean")).reset_index()


def recalibration(ns=(500, 2_000, 10_000, 50_000), prevalences=(0.5, 0.05), scenarios=tuple(SCENARIOS),
                  reps=(100, 100, 60, 30)) -> pd.DataFrame:
    """Fit every method on a calibration set of size n; score it exactly, and by ECE on the calibration set
    itself and on a fresh sample of the same size."""
    rows = []
    for (sc, prev), i in zip(itertools.product(scenarios, prevalences), itertools.count()):
        w = world(sc, prev)
        for n, r in zip(ns, reps):
            rng = np.random.default_rng([SEED, 2, i, n])
            for rep in range(r):
                y, s = w.sample(rng, n)
                y2, s2 = w.sample(rng, n)
                fits = {"none": lambda v: v, **{name: fit(s, y) for name, fit in METHODS.items()}}
                for name, q in fits.items():
                    t = true_errors(w, q)
                    rows.append((sc, prev, n, rep, name, t["l1"], t["l2"], t["excess_logloss"],
                                 ece(q(s), y, 15, "mass"), ece(q(s2), y2, 15, "mass")))
    return pd.DataFrame(rows, columns=["scenario", "prevalence", "n", "rep", "method", "true_l1", "true_l2",
                                       "excess_logloss", "ece_in_sample", "ece_fresh"])


def summarise_recalibration(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["scenario", "prevalence", "n", "method"], sort=False)
            [["true_l1", "true_l2", "excess_logloss", "ece_in_sample", "ece_fresh"]].mean().reset_index())


def auc_ties(ns=(1_000, 10_000), scenarios=("overconfident", "wiggly"), prevalence=0.5, n_test=200_000,
             reps=30, methods=("Platt", "isotonic", "histogram binning", "spline")) -> pd.DataFrame:
    """AUC on a large test sample after calibration, minus the AUC of the raw scores (ties lower it)."""
    rows = []
    for (sc, n), i in zip(itertools.product(scenarios, ns), itertools.count()):
        w = world(sc, prevalence)
        rng = np.random.default_rng([SEED, 3, i])
        for rep in range(reps):
            y, s = w.sample(rng, n)
            yt, st = w.sample(rng, n_test)
            raw = roc_auc_score(yt, st)
            for name in methods:
                qt = METHODS[name](s, y)(st)
                rows.append((sc, n, rep, name, raw, roc_auc_score(yt, qt) - raw, len(np.unique(qt))))
    return pd.DataFrame(rows, columns=["scenario", "n", "rep", "method", "raw_auc", "auc_change",
                                       "distinct_values"])


def summarise_ties(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["scenario", "n", "method"], sort=False)
            .agg(raw_auc=("raw_auc", "mean"), auc_change=("auc_change", "mean"),
                 worst_change=("auc_change", "min"), distinct_values=("distinct_values", "median"))
            .reset_index())
