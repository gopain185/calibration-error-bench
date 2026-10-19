## What the simulation shows

All numbers come from the tables above. Calibration errors are L1 (mean |predicted - true probability|) unless
the row says L2.

**Measuring calibration (ECE)**

- **ECE has a noise floor that looks like miscalibration.** For a perfectly calibrated model at a positive rate of
  0.5, the 15-bin ECE reports 0.036 on 1,000 rows, 0.012 on 10,000 and 0.004 on 100,000. With 100 bins it reports
  0.094 on 1,000 rows. A small ECE on a small test set says almost nothing.
- **The debiased L2 estimator (Kumar et al. 2019) removes most of that floor.** On the same calibrated scores it
  reports 0.011, 0.003 and 0.001, where the plain binned L2 reports 0.046, 0.015 and 0.005. On miscalibrated
  scores it errs the other way: bins average errors out, so it is low. At a positive rate of 0.5 it is within 7%
  of the true L2 error from 1,000 rows on; at 0.05 it is up to 18% low on 1,000 rows and up to 11% low on 10,000 (38% and 21% in the wiggly scenario).
- **Once miscalibration is real and the sample is large, plain binned ECE is fine.** From 10,000 rows on, the
  15-bin equal-mass ECE is within 7% of the truth in every scenario except one (below).
- **Bins hide wiggles.** When the calibration error changes sign along the score range ("wiggly", prevalence
  0.05), each bin averages positive and negative errors. Even on 100,000 rows, the equal-width ECE then reports
  0.0089 for a true 0.0132 (33% low), and the debiased L2 is 20% low. More data does not fix that; more bins do, given
  enough rows (100 equal-mass bins: 0.0136).
- The kernel ECE (Nadaraya-Watson on the logit scale) has a lower noise floor than binned ECE on small samples
  (0.023 against 0.036 on 1,000 calibrated rows). From 10,000 rows on it is within 10% of the truth, except on the
  wiggly scenario at prevalence 0.05 (11-18% low).

**Fixing calibration**

- **The right parametric family beats isotonic regression by far.** When the distortion is a pure change of
  scale, temperature scaling reaches an L1 error of 0.011 from 500 rows. Isotonic regression is at 0.044, and it
  needs about 50,000 rows to get to 0.009.
- **But each family has a blind spot that never closes.** At a positive rate of 0.5 and 50,000 rows: temperature
  cannot undo a shift (0.072, no better than doing nothing), Platt cannot undo a kink (0.032), and none of the
  three parametric methods can undo a wiggle (0.024).
- **Beta calibration handles asymmetric distortions** (kinked: 0.010 against Platt's 0.032 at 50,000 rows) and
  costs little elsewhere (10-35% more error than Platt when Platt's family is right).
- **A spline on the logit is the safest flexible method.** When a parametric family fits, the spline has 2-3
  times its error (0.015 against 0.005 for temperature on 2,000 calibrated rows), but it is the only method that
  does well in every scenario, and it wins on the kinked and wiggly ones (0.006 at 50,000 rows, prevalence 0.5).
  On 500 rows it is worse than Platt everywhere except the kinked scenario at prevalence 0.5.
- **Histogram binning never converges.** With 15 bins its error stays at about 0.017 (prevalence 0.5) however
  much data it gets, because it maps a whole range of scores to one value.
- **Isotonic regression and binning are dangerous for log loss on small data.** They output exactly 0 or 1 at
  the ends of the range, and their excess log loss on 500 rows is 0.09, against 0.003 for Platt.

**Reporting calibration**

- **ECE on the calibration set is optimistic for flexible methods.** On 500 rows, histogram binning reports an
  in-sample ECE of 0.007 while its true error is 0.040 and its ECE on fresh data is 0.045. Isotonic regression
  reports 0.021 against a true 0.031. Always measure calibration on rows the calibrator did not see.

**Calibration and AUC**

- Platt and temperature are strictly increasing maps, so the AUC is unchanged. The spline fit is not constrained
  to be monotone, but it lost at most 0.00016 AUC.
- **Isotonic regression and histogram binning create ties and lower the AUC.** After calibrating on 1,000 rows,
  isotonic regression loses 0.0024 AUC on average (up to 0.0045) and 15-bin histogram binning 0.005 (up to 0.012).
  On 10,000 rows the losses are 0.0005 and 0.0015. For a competition scored by AUC, never submit isotonic- or
  bin-calibrated predictions.

What to do with it:

- Report calibration with a debiased or kernel estimator, on held-out rows, and give the sample size.
- To recalibrate, try temperature or Platt first, then beta, then a logit spline once you have a few thousand
  rows. Use isotonic regression only with plenty of data, and not before an AUC-scored submission.
