# Calibration Error Simulations

Simulated classifier scores whose **true calibration map is known exactly**, together with the result tables of a
benchmark of calibration-error estimators (ECE and friends) and post-hoc calibration methods (temperature, Platt,
beta, isotonic, histogram binning, spline).

## Provenance

Everything is simulated by the `calbench` package (repository `calibration-error-bench` by Loc Nguyen, MIT), seed
`20261020`. The companion notebook embeds the full source and regenerates every table. No real people or records
are involved.

## How the data is generated

Each row has a true log-odds `l ~ N(mu, 2^2)` and a label `y ~ Bernoulli(sigmoid(l))`. `mu` is set so that the
positive rate is 0.5 or 0.05. A model reports the score `s = sigmoid(h(l))`, where `h` is a strictly increasing
distortion. Since `s` is a monotone function of `l`, the true calibration map is
`P(y = 1 | s) = sigmoid(h^-1(logit s))`, which is stored for every row as `true_prob`.

| scenario | h(l) |
| --- | --- |
| calibrated | l |
| overconfident | 1.6 l |
| underconfident | 0.6 l |
| shifted | l + 0.5 |
| kinked | 0.8 l below 0, 1.6 l above |
| wiggly | l + 0.6 sin(l) |

## Files

**`calibration_samples.csv`**: 10,000 rows for each scenario and prevalence (120,000 rows).

| column | meaning |
| --- | --- |
| scenario | one of the six scenarios above |
| prevalence | target positive rate, 0.5 or 0.05 |
| row_id | 0 to 9,999 within a scenario and prevalence |
| y | label, 0 or 1 |
| score | the model's score s |
| true_prob | the true calibrated probability P(y = 1 \| s) |

**`exp1_estimator_bias.csv`**: one row per replicate and estimator. Columns: scenario, prevalence, n (sample
size), positives, rep, norm (`l1` or `l2`), estimator, truth (the exact calibration error of the raw scores) and
estimate.

**`exp2_recalibration.csv`**: one row per replicate and calibration method fitted on n rows. Columns: scenario,
prevalence, n, rep, method, true_l1, true_l2 and excess_logloss (all exact, computed by integrating over the true
distribution), ece_in_sample (15 equal-mass bins, on the calibration rows) and ece_fresh (the same ECE on a new
sample of n rows).

**`exp3_auc_ties.csv`**: AUC on 200,000 new rows after calibration, minus the AUC of the raw scores. Columns:
scenario, n, rep, method, raw_auc, auc_change and distinct_values (the number of different calibrated values).

## Intended use

- Testing an implementation of ECE or any other calibration metric against the true value.
- Comparing calibration methods, or checking how much a calibration-set size buys.
- Teaching: the binned ECE is biased and the size of the bias depends on the sample size.

## Known limitations

- The true log-odds are normal and the distortions are smooth and monotone. Real models can have scores that are
  not a monotone function of the true probability, and then no calibration map recovers it.
- Only binary classification. Multi-class (top-label or class-wise) calibration is not covered.
- `true_prob` is computed by inverting `h` on a grid of 20,001 points; it is accurate to about 1e-6.

## License

CC0 1.0 (public domain).
