# Stage 3 — Qualifying uncertainty experiment

Retrospective walk-forward comparison, 2023–2026. Profile: **backtest**. Each forecast integrates 256 qualifying scenarios and 4,096 race simulations.

Production data and models are unchanged. No candidate is automatically promoted.

## Race predictions

| Candidate | Races | Spearman ↑ | Top 3 ↑ | Top 10 ↑ | Pos. MAE ↓ | Pos. log loss ↓ | Winner log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct ensemble (baseline) | 88 | 0.6226 | 56.1% | 76.4% | 3.692 | 2.7881 | 1.4278 |
| Direct LambdaRank anchor | 88 | 0.6239 | 56.1% | 76.4% | 3.671 | 2.8363 | 1.4278 |
| Basic practice only (LambdaRank) | 88 | 0.6229 | 55.3% | 76.7% | 3.693 | 2.8367 | 1.4123 |
| Qualifying-grid mixture ensemble | 88 | 0.6207 | 55.3% | 76.5% | 3.704 | 2.6610 | 1.4581 |
| Qualifying-grid mixture anchor | 88 | 0.6210 | 54.5% | 76.7% | 3.698 | 2.6778 | 1.4581 |

Top-three/top-ten accuracy means set overlap, not exact ordered finishes. Winner log loss is race-level categorical loss, not per-driver binary log loss. The JSON includes binary win/podium/points/DNF log losses and Brier scores, categorical position Brier scores, and per-season results.

## Paired uncertainty

Grid-mixture ensemble minus direct ensemble; approximate paired four-race moving-block bootstrap, 2,000 replicates. Negative loss/MAE deltas are better.

| Metric | Mean delta | 95% interval |
|---|---:|---:|
| spearman | -0.0019 | [-0.0078, +0.0034] |
| position_log_loss | -0.1271 | [-0.1784, -0.0798] |
| winner_log_loss | +0.0303 | [-0.0664, +0.1209] |
| position_mae | +0.0126 | [-0.0320, +0.0615] |

## Qualifying model

Scored 85 races with complete archived qualifying labels: Spearman **0.7351**, MAE **3.077**, pole accuracy **43.5%**, pole log loss **1.8298**, position log loss **2.6992**.

The qualifying ranker sees pre-Q inputs only. Conditional race models learn P(race | pre-Q features, historical qualifying order) on earlier races. At inference they receive sampled full-grid permutations, never the actual test Q. Monte Carlo averaging propagates grid uncertainty instead of plugging in a mean grid.

No generated qualifying feature is used during race-model training, so no in-sample stacked prediction can leak in. All qualifying forecasts used for temperature calibration are chronological out-of-fold forecasts.

## Protocol

- Identical held-out races, training support, practice recipe and calibrated pre-Q DNF probabilities.
- Retrain each race on strictly earlier race dates; earlier warmup races seed calibration.
- Qualifying and race temperatures use only earlier OOF forecasts; each fold updates state after freezing predictions.
- Conditional race training observes Q in earlier races only. Pre-Q inference marginalizes sampled full-grid permutations.
- No measured Q1/2/3 times, final starting grid, or current actual qualifying enter pre-Q scoring.
- No generated qualifying covariates are used in race-model training; no in-sample stacked predictions.
- Uniform 0.1% matrix smoothing preserves all grid probability constraints and avoids finite-MC zero losses.

Temperature choices and DNF calibration update after each forecast is frozen. Shared chronological folds and DNF probabilities isolate the grid/ranking changes. The basic-practice ablation should be compared with the direct **ranker**, not the ensemble, to isolate detailed practice features.

## Feature importance

Final held-out fold models only; training ends before that race. Global gain importance is descriptive, not causal and not per-driver SHAP.

### Qualifying

| Feature | Gain share |
|---|---:|
| driver_quali_mean_10 | 37.8% |
| constructor_points_mean_5 | 10.9% |
| driver_grid_mean_10 | 10.1% |
| practice_fp2_gap_pct | 2.7% |
| driver_season_rank_pre | 2.4% |
| practice_fp2_position_pct | 2.3% |
| driver_points_mean_10 | 2.0% |
| driver_quali_mean_5 | 1.9% |
| constructor_quali_mean_5 | 1.8% |
| practice_fp1_gap_pct | 1.7% |

### Direct

| Feature | Gain share |
|---|---:|
| driver_points_mean_10 | 23.7% |
| constructor_points_mean_10 | 12.4% |
| driver_quali_mean_10 | 8.8% |
| constructor_finish_mean_10 | 5.6% |
| driver_points_mean_5 | 3.7% |
| constructor_points_mean_5 | 3.5% |
| driver_grid_mean_10 | 2.4% |
| driver_points_mean_3 | 2.4% |
| driver_season_rank_pre | 2.1% |
| practice_fp2_position_pct | 1.9% |

### Conditional

| Feature | Gain share |
|---|---:|
| scenario_quali_percentile | 47.3% |
| scenario_quali_position | 15.0% |
| constructor_quali_mean_10 | 3.5% |
| constructor_points_mean_10 | 3.4% |
| driver_points_mean_10 | 3.0% |
| driver_points_mean_5 | 2.0% |
| constructor_finish_mean_10 | 1.6% |
| driver_quali_mean_10 | 1.1% |
| driver_points_mean_3 | 0.9% |
| practice_fp1_gap_pct | 0.8% |

## Decision

This experiment does not establish a robust improvement in both primary losses. Keep production frozen. Full deployment-profile checks and prospective forecasts are still required before any promotion.

## Limitations

- Retrospective, conditional on the archived race-entrant set; does not predict entry changes or DNQ.
- Qualifying order is not a penalty-adjusted starting grid; later penalties are not modeled.
- Historical publication/revision times are not reconstructed; 2026 is reused retrospective validation.
- A smaller development tree profile is not proof of improvement in the full production configuration.
- Four-race block bootstrap is approximate and not adjusted for repeated research/model selection.

## Artifacts

- `predictions.parquet`: all held-out full-grid race distributions and labels.
- `qualifying_predictions.parquet`: full qualifying distributions and labels.
- `cutoffs.csv`: every training cutoff, including calibration warmup races.
- `race_scores.csv`, `qualifying_scores.csv`, `metrics.json`: reproducible scoring.
- `calibration.png`, `qualifying_calibration.png`: reliability plots.
- `importance_*.csv`: all feature-gain shares.

Method references: [LightGBM LambdaRank API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) and [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
