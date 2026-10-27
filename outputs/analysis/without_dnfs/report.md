# Model accuracy with actual DNFs excluded

This diagnostic uses the leakage-safe 66-race walk-forward predictions. Drivers recorded as actual DNFs are removed after the race, then the predicted and actual orders are reranked among the remaining classified finishers.

This is an oracle diagnostic: it measures pace/order quality after retirement uncertainty is removed, but cannot be reproduced prospectively because the identities of future DNFs are unknown.

| Mode | Score | Normal | Actual DNFs excluded | Change |
|---|---|---:|---:|---:|
| pre-quali | Spearman ↑ | 0.632 | 0.743 | +0.111 |
| pre-quali | Top-3 accuracy ↑ | 56.6% | 58.6% | +2.0 pp |
| pre-quali | Top-10 accuracy ↑ | 75.6% | 80.9% | +5.3 pp |
| pre-quali | Winner accuracy ↑ | 30.3% | 34.8% | +4.5 pp |
| pre-quali | Position MAE ↓ | 3.65 | 2.66 | -0.99 |
| pre-quali | Drivers scored | 1,354 | 1,180 | 174 DNFs removed |
| post-quali | Spearman ↑ | 0.694 | 0.801 | +0.107 |
| post-quali | Top-3 accuracy ↑ | 67.7% | 70.2% | +2.5 pp |
| post-quali | Top-10 accuracy ↑ | 78.9% | 83.9% | +5.0 pp |
| post-quali | Winner accuracy ↑ | 57.6% | 63.6% | +6.1 pp |
| post-quali | Position MAE ↓ | 3.20 | 2.23 | -0.97 |
| post-quali | Drivers scored | 1,354 | 1,180 | 174 DNFs removed |

## Austin 2026

Austin had 3 recorded DNFs. After removing those drivers, Spearman changes from 0.919 to 0.984, and position MAE changes from 1.64 to 0.74. Top-three and top-ten overlap remain 100% and 100%.

## Interpretation

The classified-only score separates the model's pace/order skill from the harder task of predicting retirements. The deployable forecast must still retain DNF probability and all drivers; the normal columns remain the honest end-to-end performance numbers.
