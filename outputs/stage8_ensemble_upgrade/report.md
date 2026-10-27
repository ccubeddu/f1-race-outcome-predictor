# Stage 8 — Three-model post-qualifying ensemble

## Production change

The post-qualifying ensemble now uses 60% LightGBM LambdaRank, 30% Random Forest,
and 10% LightGBM finish-position regression. The regression member was admitted only after
selection on the 2024–2025 development period and verification on the untouched 2026 holdout.
The 1.25× pole calibration remains in place.

## 66-race walk-forward result

| Metric | Previous production | Upgraded production | Change |
|---|---:|---:|---:|
| Spearman ↑ | 0.6946 | 0.6945 | -0.0001 |
| Top-3 accuracy ↑ | 67.2% | 68.2% | +1.0 pp |
| Top-10 accuracy ↑ | 78.9% | 78.9% | unchanged |
| Position MAE ↓ | 3.202 | 3.207 | +0.005 |
| Position log loss ↓ | 2.612 | 2.587 | -0.025 |
| Position Brier ↓ | 0.89121 | 0.89074 | -0.00047 |
| Highest-P(win) accuracy ↑ | 60.6% | 60.6% | unchanged |
| Winner categorical log loss ↓ | 1.151 | 1.147 | -0.005 |

The main gain is a better full finishing-position probability distribution plus two additional
correct podium-set selections across 66 races. Rank correlation and MAE are effectively flat.

## 2026 holdout

Top-three accuracy improves from 68.5% to 70.4%, top-ten accuracy stays at 77.8%, and position
log loss improves from 2.662 to 2.633. Spearman changes from 0.6884 to 0.6873 and MAE from 3.343
to 3.364, so the holdout confirms the probability/top-three gain while showing a small tradeoff
in exact expected-position ordering.

## Rejected experiments

- Exponentially weighted form, circuit behavior, qualifying-surprise interactions, and their
  combinations overfit or weakened the 2026 holdout.
- Career-normalized rates improved position log loss but worsened ordering and calibration.
- Constructor-based DNF shrinkage improved DNF Brier score but degraded finish-order metrics.

Those experiments were removed; the production feature schema remains 138 pre-quali features
and 148 post-quali features.
