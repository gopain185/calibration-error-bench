# calibration-error-bench

How far is a reported ECE from the real calibration error, and which post-hoc calibration method actually
recovers the true probabilities? On real data nobody knows a row's true probability, so this repository
simulates classifier scores whose **true calibration map is known exactly** and measures both.

- **Kaggle notebook:** [How Wrong Is Your ECE? Calibration, Known Truth](https://www.kaggle.com/code/locdunhat/how-wrong-is-your-ece-calibration-known-truth)
- **Kaggle dataset:** [Calibration Error Simulations](https://www.kaggle.com/datasets/locdunhat/calibration-error-simulations) (CC0)

## The model

Each row has a true log-odds `l ~ N(mu, 2^2)` and a label `y ~ Bernoulli(sigmoid(l))`, with `mu` set for a
positive rate of 0.5 or 0.05. The model reports `s = sigmoid(h(l))` for a strictly increasing distortion `h`.
Because `s` is a monotone function of `l`, the true map is `c(s) = P(y = 1 | s) = sigmoid(h^-1(logit s))`, and the
exact calibration error of any recalibrated score `q(s)` is an integral over `l` (`calbench/sim.py`).

| scenario | h(l) |
| --- | --- |
| calibrated | l |
| overconfident | 1.6 l |
| underconfident | 0.6 l |
| shifted | l + 0.5 |
| kinked | 0.8 l below 0, 1.6 l above |
| wiggly | l + 0.6 sin(l) |

Estimators (`calbench/estimators.py`): binned ECE with 15 equal-width, 15 equal-mass or 100 equal-mass bins, a
kernel ECE (Nadaraya-Watson on the logit scale), and the binned L2 error, plain and debiased (Kumar, Liang & Ma
2019).

Calibrators (`calbench/calibrators.py`): temperature scaling, Platt scaling, beta calibration (Kull et al. 2017),
isotonic regression, histogram binning (15 equal-mass bins) and logistic regression on a cubic spline of the logit.

## Results

**1. ECE of a perfectly calibrated model** (true error 0; positive rate 0.5; mean over 300 replicates, 100 at 100k):

| rows | ECE, 15 equal-mass bins | ECE, 100 bins | kernel ECE | L2, 15 bins | L2 debiased |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 0.036 | 0.094 | 0.023 | 0.046 | 0.011 |
| 10,000 | 0.011 | 0.030 | 0.009 | 0.015 | 0.003 |
| 100,000 | 0.004 | 0.009 | 0.004 | 0.005 | 0.001 |

On miscalibrated scores, the 15-bin equal-mass ECE is within about 7% of the truth from 10,000 rows on, except
when the error changes sign along the score range ("wiggly"): bins then average it out, and even 100,000 rows
give an equal-width ECE 33% too low. The debiased L2 removes the noise floor but is up to 18% low on
miscalibrated scores at a positive rate of 0.05.

**2. Exact L1 calibration error after recalibration** (positive rate 0.5):

| scenario | rows | none | temperature | Platt | beta | isotonic | hist. binning | spline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overconfident | 500 | 0.066 | **0.011** | 0.019 | 0.022 | 0.045 | 0.055 | 0.032 |
| overconfident | 50,000 | 0.066 | **0.0013** | 0.0022 | 0.0025 | 0.0092 | 0.017 | 0.0041 |
| shifted | 50,000 | 0.075 | 0.072 | **0.0019** | 0.0022 | 0.0090 | 0.017 | 0.0036 |
| kinked | 50,000 | 0.050 | 0.052 | 0.032 | 0.010 | 0.0091 | 0.017 | **0.0058** |
| wiggly | 50,000 | 0.049 | 0.024 | 0.024 | 0.024 | 0.0090 | 0.017 | **0.0060** |

- The right parametric family wins by far, but each one has a blind spot that more data never closes.
- Histogram binning stalls at about 0.017. Isotonic regression needs tens of thousands of rows.
- Measured on the calibration set itself, histogram binning reports an ECE of 0.007 on 500 rows while its true
  error is 0.040.

**3. AUC lost to ties** (200,000 test rows): calibrating on 1,000 rows costs 0.0024 AUC with isotonic regression
and 0.005 with histogram binning, and 0.0005 / 0.0015 on 10,000 rows. Platt and temperature cost nothing.

The full tables and more findings are in the notebook.

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q                              # 15 tests, about 5 s
python -m calbench results                       # all experiments, about 35 minutes on CPU
python kaggle/build_kaggle.py --reuse-results    # Kaggle dataset and notebook from results/
```

```python
from calbench.estimators import ece, ece_l2_debiased, kernel_ece
from calbench.calibrators import platt, spline

print(ece(scores, labels), ece_l2_debiased(scores, labels), kernel_ece(scores, labels))
q = spline(cal_scores, cal_labels)       # returns a function
calibrated = q(test_scores)
```

## Layout

```
calbench/sim.py           true log-odds, score distortions, exact calibration errors
calbench/estimators.py    binned, debiased and kernel calibration-error estimators
calbench/calibrators.py   temperature, Platt, beta, isotonic, histogram binning, spline
calbench/experiments.py   the three experiments and their summaries
tests/                    checks of the true map, the exact errors, the estimators and the calibrators
kaggle/                   builder, datasheet and text of the Kaggle notebook
results/                  raw tables of the last full run
```

## Limitations

- The true log-odds are normal and the distortions are monotone. A model whose scores are not a monotone
  function of the true probability cannot be fixed by any calibration map, and that case is not covered.
- Binary classification only.
- The kernel ECE is a simple Nadaraya-Watson estimate with Silverman's bandwidth, not the smooth ECE of Błasiok
  and Nakkiran (2023).

## License

MIT. Author: Loc Nguyen.
