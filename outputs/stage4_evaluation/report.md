# Walk-forward evaluation

Every test race was predicted by models trained only on earlier races. Probability calibration used only earlier out-of-fold predictions.

| Mode | Model | Spearman | Top 3 | Top 10 | Position MAE | Position log loss | DNF Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| pre_quali | LightGBM LambdaRank | 0.626 | 56.1% | 75.6% | 3.70 | 2.893 | 0.112 |
| pre_quali | LightGBM regression | 0.632 | 50.0% | 77.3% | 3.67 | 3.511 | 0.112 |
| pre_quali | Random Forest | 0.633 | 52.0% | 75.9% | 3.66 | 2.896 | 0.112 |
| post_quali | LightGBM LambdaRank | 0.686 | 67.2% | 78.8% | 3.26 | 2.697 | 0.111 |
| post_quali | LightGBM regression | 0.690 | 59.6% | 78.9% | 3.29 | 3.135 | 0.111 |
| post_quali | Random Forest | 0.687 | 60.1% | 78.5% | 3.29 | 2.650 | 0.111 |

Top-3 and top-10 accuracy are set-overlap recall within each race. Position log loss scores the probability assigned to the exact observed finishing position.

## LambdaRank probability quality

| Mode | Outcome | Log loss | Brier | Null Brier | Brier skill |
|---|---|---:|---:|---:|---:|
| pre_quali | dnf | 0.382 | 0.112 | 0.112 | 0.3% |
| pre_quali | podium | 0.260 | 0.082 | 0.125 | 33.9% |
| pre_quali | points | 0.509 | 0.165 | 0.250 | 33.8% |
| pre_quali | win | 0.126 | 0.039 | 0.046 | 16.6% |
| post_quali | dnf | 0.380 | 0.111 | 0.112 | 0.9% |
| post_quali | podium | 0.205 | 0.066 | 0.125 | 47.5% |
| post_quali | points | 0.463 | 0.147 | 0.250 | 41.2% |
| post_quali | win | 0.093 | 0.028 | 0.046 | 39.7% |

Post-qualifying LambdaRank is the recommended model: it has the strongest top-3 accuracy and win/podium probability scores. Random Forest has slightly lower post-qualifying exact-position log loss, so it remains a useful challenger.

DNF Brier skill is close to zero. These probabilities are base-rate calibrated but have weak discrimination; improving component-level reliability data is a priority.
