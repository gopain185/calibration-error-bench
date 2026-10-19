import numpy as np
import pytest
from scipy.special import logit

from calbench.calibrators import METHODS, histogram, isotonic, platt, temperature
from calbench.estimators import ece, ece_l2, ece_l2_debiased, kernel_ece
from calbench.experiments import auc_ties, estimator_bias, recalibration
from calbench.sim import SCENARIOS, Distortion, true_errors, world


@pytest.mark.parametrize("prevalence", [0.5, 0.05])
def test_prevalence(prevalence):
    y, _ = world("shifted", prevalence).sample(np.random.default_rng(0), 400_000)
    assert abs(y.mean() - prevalence) < 0.003


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_true_map_matches_label_rates(scenario):
    w = world(scenario, 0.3)
    y, s = w.sample(np.random.default_rng(1), 1_000_000)
    c = w.true_map(s)
    b = np.minimum((np.argsort(np.argsort(s)) * 20) // len(s), 19)  # 20 equal-mass bins
    rate = np.bincount(b, y) / np.bincount(b)
    mean_c = np.bincount(b, c) / np.bincount(b)
    assert np.abs(rate - mean_c).max() < 0.006


def test_exact_errors():
    assert true_errors(world("calibrated"))["l1"] < 1e-12
    for sc in SCENARIOS:
        w = world(sc, 0.2)
        assert true_errors(w, w.true_map)["l1"] < 1e-4  # recalibrating with the true map removes the error
    w = world("kinked", 0.2)
    y, s = w.sample(np.random.default_rng(2), 1_000_000)
    assert true_errors(w)["l1"] == pytest.approx(np.abs(s - w.true_map(s)).mean(), rel=0.01)


def test_distortion_must_increase():
    with pytest.raises(ValueError):
        Distortion("bad", a=1.0, wiggle=1.0)


def test_ece_by_hand():
    s = np.array([0.05, 0.15, 0.95, 0.85])
    y = np.array([0, 1, 1, 1])
    # equal width: bins [0.0, 0.0667) {0.05}, [0.1333, 0.2) {0.15}, [0.8, 0.8667) {0.85}, [0.9333, 1) {0.95}
    assert ece(s, y, 15, "width") == pytest.approx((0.05 + 0.85 + 0.05 + 0.15) / 4)
    # two equal-mass bins: {0.05, 0.15} with rate 0.5, {0.85, 0.95} with rate 1
    assert ece(s, y, 2, "mass") == pytest.approx((abs(0.5 - 0.1) + abs(1 - 0.9)) / 2)
    assert ece_l2(s, y, 2, "mass") == pytest.approx(np.sqrt((0.4 ** 2 + 0.1 ** 2) / 2))


def test_estimators_on_calibrated_scores():
    w = world("calibrated")
    y, s = w.sample(np.random.default_rng(3), 50_000)
    assert ece_l2_debiased(s, y) < ece_l2(s, y)
    assert ece_l2_debiased(s, y) < 0.004
    w = world("overconfident")
    y, s = w.sample(np.random.default_rng(4), 200_000)
    assert kernel_ece(s, y) == pytest.approx(true_errors(w)["l1"], rel=0.05)


def test_calibrators_recover_simple_distortions():
    w = world("overconfident")
    y, s = w.sample(np.random.default_rng(5), 100_000)
    assert true_errors(w, platt(s, y))["l1"] < 0.003
    q = temperature(s, y)
    assert logit(q(np.array([0.9])))[0] == pytest.approx(logit(0.9) / 1.6, rel=0.03)
    qi = isotonic(s, y)(np.sort(s))
    assert np.all(np.diff(qi) >= 0)
    assert len(np.unique(histogram(s, y)(s))) <= 15


def test_every_method_beats_raw_on_large_samples():
    for sc in ["overconfident", "underconfident", "shifted"]:
        w = world(sc)
        y, s = w.sample(np.random.default_rng(6), 50_000)
        raw = true_errors(w)["l1"]
        for name, fit in METHODS.items():
            after = true_errors(w, fit(s, y))["l1"]
            if sc == "shifted" and name == "temperature":
                assert after > raw * 0.8  # a single scale cannot undo a shift
            else:
                assert after < raw / 2, (sc, name)


def test_experiments_run_small():
    e1 = estimator_bias(ns=(500,), prevalences=(0.5,), scenarios=("shifted",), reps=(3,))
    assert len(e1) == 3 * 6 and e1.truth.nunique() == 2
    e2 = recalibration(ns=(500,), prevalences=(0.5,), scenarios=("wiggly",), reps=(2,))
    assert set(e2.method) == {"none", *METHODS}
    e3 = auc_ties(ns=(500,), scenarios=("wiggly",), n_test=5_000, reps=2)
    assert (e3.auc_change <= 1e-12).all()  # a monotone map cannot raise the AUC
