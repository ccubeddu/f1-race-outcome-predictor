# Walk-forward evaluation

Every test race was predicted by models trained only on earlier races. Probability calibration used only earlier out-of-fold predictions.

| Mode | Model | Spearman | Top 3 | Top 10 | Position MAE | Position log loss | DNF Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| post_quali | Anchored ensemble | 0.694 | 68.2% | 78.9% | 3.21 | 2.589 | 0.111 |
| post_quali | LightGBM LambdaRank | 0.686 | 66.2% | 78.6% | 3.26 | 2.669 | 0.111 |
| post_quali | LightGBM regression | 0.688 | 61.1% | 79.1% | 3.31 | 3.107 | 0.111 |
| post_quali | Pole-aware ensemble | 0.694 | 68.2% | 78.9% | 3.21 | 2.587 | 0.111 |
| post_quali | Random Forest | 0.691 | 60.6% | 78.6% | 3.27 | 2.639 | 0.111 |

Top-3 and top-10 accuracy are set-overlap recall within each race. Position log loss scores the probability assigned to the exact observed finishing position.

## Production ensemble probability quality

| Mode | Outcome | Log loss | Brier | Null Brier | Brier skill |
|---|---|---:|---:|---:|---:|
| post_quali | dnf | 0.380 | 0.111 | 0.112 | 0.9% |
| post_quali | podium | 0.206 | 0.066 | 0.125 | 47.5% |
| post_quali | points | 0.462 | 0.147 | 0.250 | 41.3% |
| post_quali | win | 0.092 | 0.028 | 0.046 | 40.5% |

The anchored ensemble blends structurally different rankers for finishing order. Post-qualifying predictions then apply a holdout-validated pole adjustment to P(win) while preserving P(podium), P(points), and joint race coherence.

DNF Brier skill is close to zero. These probabilities are base-rate calibrated but have weak discrimination; improving component-level reliability data is a priority.
