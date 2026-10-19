"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only] [--reuse-results]

writes kaggle/dataset/ (samples of scores with their true calibrated probability, the result tables of the three
experiments and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and reruns the
experiments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calbench.experiments import SEED, auc_ties, estimator_bias, recalibration  # noqa: E402
from calbench.sim import SCENARIOS, world  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/calibration-error-simulations"
NOTEBOOK_ID = "locdunhat/how-wrong-is-your-ece-calibration-known-truth"
TITLE = "How Wrong Is Your ECE? Calibration, Known Truth"
ROWS = 10_000


def build_dataset(public: bool, rerun: bool = True) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (sc, prev) in enumerate((sc, prev) for sc in SCENARIOS for prev in (0.5, 0.05)):
        w = world(sc, prev)
        y, s = w.sample(np.random.default_rng([SEED, 0, i]), ROWS)
        parts.append(pd.DataFrame({"scenario": sc, "prevalence": prev, "row_id": np.arange(ROWS), "y": y,
                                   "score": s.round(8), "true_prob": w.true_map(s).round(8)}))
    pd.concat(parts).to_csv(out / "calibration_samples.csv", index=False)
    if rerun:
        for name, fn in [("exp1_estimator_bias", estimator_bias), ("exp2_recalibration", recalibration),
                         ("exp3_auc_ties", auc_ties)]:
            fn().to_csv(out / f"{name}.csv", index=False)
    else:  # reuse the tables written by `python -m calbench results`
        for f in sorted((ROOT / "results").glob("exp*.csv")):
            (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Calibration Error Simulations",
        "id": DATASET_ID,
        "subtitle": "Scores with a known calibration map, for testing ECE and calibrators",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["tabular", "benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "calbench").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


SETUP = """

import os, sys
ROOT = "/kaggle/working/calibration-error-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("calibration-error-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import expit
from calbench import experiments as ex
from calbench.sim import SCENARIOS, world
pd.set_option("display.precision", 4); pd.set_option("display.width", 160)
"""

MAPS = """fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for prev, ax in zip([0.5, 0.05], axes):
    for sc in SCENARIOS:
        w = world(sc, prev)
        s = expit(np.linspace(w.distortion.h(np.array([w.mu - 3 * w.sigma]))[0], w.distortion.h(np.array([w.mu + 3 * w.sigma]))[0], 400))
        ax.plot(s, w.true_map(s), label=sc)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_title(f"true calibration maps, prevalence {prev}")
    ax.set_xlabel("model score s"); ax.set_ylabel("P(y = 1 | s)")
axes[0].legend(); plt.tight_layout(); plt.show()
"""

EXP1 = """t = time.time(); e1 = ex.summarise_bias(ex.estimator_bias()); print(round(time.time() - t), "s")
e1["rel_bias"] = e1.bias / e1.truth.where(e1.truth > 0)
e1.pivot_table(index=["scenario", "prevalence", "n"], columns="estimator", values="mean_estimate", sort=False).join(
    e1.groupby(["scenario", "prevalence", "n"], sort=False).truth.first().rename("TRUE l1 / l2"))
"""

EXP1B = """e1.pivot_table(index=["scenario", "prevalence", "n"], columns="estimator", values="rmse", sort=False)
"""

EXP2 = """t = time.time(); e2 = ex.summarise_recalibration(ex.recalibration()); print(round(time.time() - t), "s")
e2.pivot_table(index=["scenario", "prevalence", "n"], columns="method", values="true_l1", sort=False)
"""

EXP2B = """e2.pivot_table(index=["scenario", "prevalence", "n"], columns="method", values="excess_logloss", sort=False)
"""

EXP2C = """e2[e2.method.isin(["isotonic", "histogram binning", "Platt", "spline"])].pivot_table(
    index=["method", "n"], values=["true_l1", "ece_in_sample", "ece_fresh"], sort=False)
"""

EXP3 = """t = time.time(); e3 = ex.summarise_ties(ex.auc_ties()); print(round(time.time() - t), "s"); e3
"""

SAMPLE = """import glob
from calbench.estimators import L1
from calbench.calibrators import METHODS
hits = glob.glob("/kaggle/input/**/calibration_samples.csv", recursive=True)
if hits:
    df = pd.read_csv(hits[0])
    rows = []
    for (sc, prev), g in df.groupby(["scenario", "prevalence"], sort=False):
        y, s, c = g.y.to_numpy(), g.score.to_numpy(), g.true_prob.to_numpy()
        rows.append({"scenario": sc, "prevalence": prev, "sample |s - c(s)|": np.abs(s - c).mean(),
                     **{k: fn(s, y) for k, fn in L1.items()}})
    display(pd.DataFrame(rows))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## The six scenarios\n\nEach curve is the true calibration map `c(s) = P(y = 1 | s)` of a "
                     "scenario. A calibrated model sits on the diagonal."),
        ("code", MAPS),
        ("markdown", "## 1. How good are the ECE estimators?\n\nEach estimator is computed on the raw scores of a "
                     "fresh sample (300 replicates at 1k and 10k rows, 100 at 100k). The table shows the mean "
                     "estimate next to the exact value (`TRUE l1 / l2`: the L1 error for the ECE columns, the L2 "
                     "error for the two L2 columns)."),
        ("code", EXP1),
        ("markdown", "Root mean squared error of each estimator against the truth:"),
        ("code", EXP1B),
        ("markdown", "## 2. Which calibration method recovers the truth?\n\nEach method is fitted on a calibration "
                     "set of `n` rows and scored exactly against the true map (100 replicates at 500 and 2k rows, "
                     "60 at 10k, 30 at 50k). `none` is the raw score. First the exact L1 calibration error:"),
        ("code", EXP2),
        ("markdown", "The excess log loss over the true probabilities (0 is perfect):"),
        ("code", EXP2B),
        ("markdown", "ECE measured on the calibration set itself, on a fresh sample of the same size, and the true "
                     "error (averaged over scenarios and prevalences):"),
        ("code", EXP2C),
        ("markdown", "## 3. Does calibration cost AUC?\n\nAUC on 200,000 fresh rows after calibrating on `n` rows, "
                     "minus the AUC of the raw scores. Isotonic regression and histogram binning map many scores "
                     "to the same value, and ties lower the AUC."),
        ("code", EXP3),
        ("markdown", "## 4. The samples in the dataset\n\nThe companion dataset holds 10,000 rows per scenario and "
                     "prevalence, with each row's true calibrated probability. Here are the estimators on them."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "calibration-error.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "calibration-error.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["evaluation", "benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public, rerun="--reuse-results" not in sys.argv))
    print("wrote", build_notebook(public))
