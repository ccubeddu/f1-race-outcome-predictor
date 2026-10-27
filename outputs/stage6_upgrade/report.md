# Stage 6 — Anchored ensemble upgrade

## Decision

The production predictor now uses an **anchored ensemble** instead of LambdaRank alone. The
ensemble combines LightGBM LambdaRank, Random Forest, and LightGBM finish-position regression.
It then uses iterative proportional fitting to preserve the LambdaRank member's calibrated
P(win), P(podium), and P(points) exactly while improving the detailed finishing-position shape.

The weights are prediction-mode specific:

- Pre-quali: 50% LambdaRank, 30% Random Forest, 20% LightGBM regression.
- Post-quali: 65% LambdaRank, 35% Random Forest.

## Full walk-forward benchmark

These are out-of-fold results over 66 races from 2024–2026. Every race was predicted using only
earlier races.

| Mode | Metric | Previous LambdaRank | Anchored ensemble | Change |
|---|---|---:|---:|---:|
| Pre-quali | Spearman ↑ | 0.626 | 0.632 | +0.006 |
| Pre-quali | Top-3 accuracy ↑ | 56.1% | 56.6% | +0.5 pp |
| Pre-quali | Top-10 accuracy ↑ | 75.6% | 75.6% | unchanged |
| Pre-quali | Position MAE ↓ | 3.70 | 3.65 | −0.05 |
| Pre-quali | Position log loss ↓ | 2.893 | 2.793 | −3.5% |
| Post-quali | Spearman ↑ | 0.686 | 0.694 | +0.008 |
| Post-quali | Top-3 accuracy ↑ | 67.2% | 67.7% | +0.5 pp |
| Post-quali | Top-10 accuracy ↑ | 78.8% | 78.9% | +0.2 pp |
| Post-quali | Position MAE ↓ | 3.26 | 3.20 | −0.06 |
| Post-quali | Position log loss ↓ | 2.697 | 2.614 | −3.1% |

## Later-season holdout

The weights were checked against the 18 completed 2026 races as a later-period holdout. The
upgrade generalizes in the metrics it is intended to improve:

| Mode | Metric | Previous LambdaRank | Anchored ensemble | Change |
|---|---|---:|---:|---:|
| Pre-quali | Spearman ↑ | 0.641 | 0.657 | +0.015 |
| Pre-quali | Top-3 accuracy ↑ | 57.4% | 59.3% | +1.9 pp |
| Pre-quali | Top-10 accuracy ↑ | 76.1% | 76.1% | unchanged |
| Pre-quali | Position MAE ↓ | 3.73 | 3.61 | −3.4% |
| Pre-quali | Position log loss ↓ | 2.800 | 2.704 | −3.4% |
| Post-quali | Spearman ↑ | 0.675 | 0.688 | +0.013 |
| Post-quali | Top-3 accuracy ↑ | 70.4% | 70.4% | unchanged |
| Post-quali | Top-10 accuracy ↑ | 77.8% | 77.8% | unchanged |
| Post-quali | Position MAE ↓ | 3.42 | 3.34 | −2.5% |
| Post-quali | Position log loss ↓ | 2.744 | 2.665 | −2.9% |

## Probability guarantees

Anchoring divides the finishing distribution into four blocks: P1, P2–P3, P4–P10, and P11+.
Within each block, the challenger models improve the allocation of exact positions. Block totals
remain equal to LambdaRank for every driver. Therefore the new model has exactly the same tested
win, podium, and points Brier/log-loss scores as the previous production model, rather than trading
calibration away for better ranking metrics.

The regenerated audit reports:

- zero chronology violations;
- maximum probability-normalization error of 5.55e-16;
- one driver per finishing position in every simulated race;
- each driver's complete position distribution sums to 100%;
- grid-wide win probability sums to 100%.

## Rejected experiment

A conservative DNF-probability shrinkage layer was tested. It did not improve both the development
period and the 2026 holdout, so it was not shipped. DNF remains the weakest target: post-quali DNF
Brier skill is 0.9% over the base rate. Improving this properly requires richer component-failure,
damage, and incident data rather than a cosmetic recalibration.

## Deployment status

Both pre- and post-quali production artifacts were retrained on all 191 available races. The live
CLI now runs every weighted member, combines their joint Monte Carlo distributions, applies the
probability constraints, and returns the upgraded coherent result. The command interface is
unchanged.
