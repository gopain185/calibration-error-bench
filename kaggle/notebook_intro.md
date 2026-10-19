# How Wrong Is Your ECE? Calibration, Known Truth

Expected calibration error (ECE) is the standard way to report whether predicted probabilities can be taken at
face value, and isotonic regression or Platt scaling the standard fix. But ECE is itself an estimate from a finite
sample. How far is it from the real calibration error, and which fix actually recovers the true probabilities?

On real data nobody knows the true probability of a row, so neither question can be answered there. This
notebook simulates scores whose **true calibration map is known exactly**, in six scenarios: calibrated,
overconfident, underconfident, shifted, kinked and wiggly, each at a positive rate of 0.5 and 0.05. Then it
measures:

1. the bias and error of six calibration-error estimators: binned ECE with 15 equal-width, 15 equal-mass or 100
   equal-mass bins, a kernel ECE, and the binned L2 error with and without the debiasing of Kumar et al. (2019);
2. how close temperature scaling, Platt scaling, beta calibration, isotonic regression, histogram binning and a
   spline fit get to the true map, from 500 to 50,000 calibration rows;
3. how much AUC isotonic regression and histogram binning lose to the ties they create.

Everything runs on CPU in about 25 minutes, and the package source is embedded below.
