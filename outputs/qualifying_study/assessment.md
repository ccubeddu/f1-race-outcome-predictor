# Stage 3 assessment

The qualifying-grid model is useful for full finishing distributions, but is not an across-the-board upgrade. Keep production unchanged.

- Position log loss improves **4.6%**: 2.7881 → 2.6610. The paired block-bootstrap interval favors the mixture, and its position log loss is lower in each of the four evaluated seasons.
- Winner log loss worsens: 1.4278 → 1.4581; the paired uncertainty interval crosses zero.
- Rank correlation is essentially flat: 0.6226 → 0.6207. Better exact-position probabilities do not imply a better ordering.

## Calibration weakness

For 254 qualifying driver-race predictions in the 10–30% pole-probability range, mean predicted probability is **15.8%**, versus **24.8%** observed pole frequency. The pole reliability curve therefore shows under-confidence in this range. Higher-probability bins are sparse and should not be overinterpreted.

A sensible next research test is a separately calibrated first-place draw followed by a conditional ordering of the remaining drivers, with both calibrators fit only to earlier OOF forecasts. That could sharpen P(pole) while retaining uncertain lower grid positions. This has not been implemented or evaluated, and this same retrospective window would not become a fresh holdout for that test.

## Practice ablation

Compare the rankers, since the ensemble also changes within-position-block ordering. Detailed versus basic practice position log loss is 2.8363 versus 2.8367; winner log loss is 1.4278 versus 1.4123. This experiment does not show a clear predictive gain from the new detailed practice features. Both use the same detailed-feature DNF model, so this is specifically a finish-order feature ablation, not an all-pipeline reliability ablation.

## Scope

The baseline is a direct ensemble **refit on the expanded stage 2 dataset**, not the frozen old production artifact. All comparisons use matched development-size models, known archived entrants and earlier-race-only fitting/calibration. Historical publication timing and entry-list availability are not reconstructed. The experiment does not establish improvement over the old production model, nor validate the full deployment tree configuration. Monte Carlo convergence across seeds and scenario counts remains untested.

The main report includes raw metrics, full probability distributions, calibration plots and global feature importance. See `audit.json` for all probability constraints and frozen production fingerprints.
