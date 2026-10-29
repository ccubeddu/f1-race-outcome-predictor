# Stage 4 — Driver, constructor and circuit ratings

Matched expanding-window pre-quali comparison, 2023–2026. Tree profile: **backtest**. Production artifacts unchanged; no candidate promoted.

## Race metrics

| Candidate | Races | Spearman ↑ | Top 3 ↑ | Top 10 ↑ | Pos. MAE ↓ | Position LL ↓ | Winner LL ↓ | DNF Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct ensemble (baseline) | 88 | 0.6226 | 56.1% | 76.4% | 3.692 | 2.7881 | 1.4278 | 0.11256 |
| Direct ranker | 88 | 0.6239 | 56.1% | 76.4% | 3.671 | 2.8363 | 1.4278 | 0.11256 |
| Global ratings ranker | 88 | 0.6245 | 53.8% | 76.4% | 3.678 | 2.8218 | 1.4760 | 0.11256 |
| Ratings + circuit ranker | 88 | 0.6214 | 54.5% | 76.0% | 3.693 | 2.8216 | 1.4223 | 0.11256 |
| Ratings + circuit ensemble | 88 | 0.6236 | 55.7% | 75.9% | 3.690 | 2.7542 | 1.4223 | 0.11256 |
| Ratings + circuit + reliability ranker | 88 | 0.6219 | 54.2% | 76.1% | 3.689 | 2.8110 | 1.4340 | 0.11248 |
| Ratings + circuit + reliability ensemble | 88 | 0.6222 | 54.5% | 75.9% | 3.693 | 2.7477 | 1.4340 | 0.11248 |

Top-3/top-10 measure set overlap. Winner LL is race-level categorical loss. The JSON also includes binary log losses/Brier scores and position Brier scores.

## Ablation protocol

- Compare direct, global-ratings and circuit-ratings **rankers** to isolate feature additions.
- Compare the direct and circuit-ratings **ensembles** for the matched model-stack comparison.
- Those models share exactly the same base DNF probabilities.
- The reliability variant reuses the circuit-rating finish models, changing only its DNF classifier inputs/calibration and downstream temperature calibration.
- All models use the same training rows and held-out races, frozen settings, simulation seeds and earlier-only calibration. The direct baseline is refit on staged data, not the old frozen production artifact.

## Paired uncertainty versus direct ensemble

Candidate minus baseline; 95% approximate four-race moving-block bootstrap intervals. Lower losses/MAE are better; higher Spearman is better.

| Candidate | Metric | Mean delta | 95% interval |
|---|---|---:|---:|
| Ratings + circuit ensemble | spearman | +0.0010 | [-0.0045, +0.0065] |
| Ratings + circuit ensemble | position_log_loss | -0.0340 | [-0.0531, -0.0168] |
| Ratings + circuit ensemble | winner_log_loss | -0.0055 | [-0.0864, +0.0706] |
| Ratings + circuit ensemble | position_mae | -0.0018 | [-0.0363, +0.0325] |
| Ratings + circuit + reliability ensemble | spearman | -0.0004 | [-0.0067, +0.0058] |
| Ratings + circuit + reliability ensemble | position_log_loss | -0.0405 | [-0.0636, -0.0203] |
| Ratings + circuit + reliability ensemble | winner_log_loss | +0.0062 | [-0.0792, +0.0843] |
| Ratings + circuit + reliability ensemble | position_mae | +0.0013 | [-0.0386, +0.0376] |

## Regulation-era breakdown

| Candidate | Era | Races | Spearman | Position LL | DNF Brier |
|---|---|---:|---:|---:|---:|
| Direct ensemble (baseline) | 2022-2025 | 70 | 0.6144 | 2.7970 | 0.10116 |
| Direct ensemble (baseline) | 2026+ | 18 | 0.6545 | 2.7538 | 0.15278 |
| Ratings + circuit ensemble | 2022-2025 | 70 | 0.6178 | 2.7607 | 0.10116 |
| Ratings + circuit ensemble | 2026+ | 18 | 0.6462 | 2.7285 | 0.15278 |
| Ratings + circuit + reliability ensemble | 2022-2025 | 70 | 0.6163 | 2.7514 | 0.10121 |
| Ratings + circuit + reliability ensemble | 2026+ | 18 | 0.6450 | 2.7332 | 0.15227 |

The JSON contains the full per-season results, not just era averages.

## Feature design and leakage policy

Rating sidecar: 7,307 driver-race rows; 17 global, 22 circuit/layout, and 20 reliability features.

- All entrants are snapshotted before any event update. Qualifying ratings also update only after the race; current qualifying is not used by these pre-Q features.
- A pairwise logistic update models driver + constructor strength. Driver updates include teammate comparisons. Race-rank updates exclude source-defined DNFs/DNS/DSQ and use classified finishers only; qualifying has its own channel.
- Driver ratings persist through moves; constructor states stay with the operation. Time-bounded Enstone/Hinwil/Faenza/Silverstone mappings share discounted latent state, never official championship points. Other IDs stay separate; unrelated Lotus teams are not merged, and Cadillac starts without predecessor evidence.
- Annual, regulation-era and rebrand discounts are fixed research priors, not fitted to these test seasons. All constants are in `feature_manifest.json`.
- Circuit-layout and coarse geometry-bucket residuals shrink sparse evidence toward zero. They do not feed back into global rating updates, keeping the feature ablation clean.
- Event-specific F1DB layout geometry is used, never today's circuit layout, actual race distance, future aggregate wins or championship-decider flags.
- Mechanical-, incident- and other-labeled DNF histories are separate from pace. Dirichlet-smoothed component rates sum to total DNF rate. Generic 'Retired' stays unknown/other; incident labels do not establish driver fault.

## Layout audit

Verified-date geometry coverage: 99.42%. Disagreeing dates are excluded from geometry attribution, using unique unknown layout keys so they cannot borrow another circuit's evidence.

- 2021 R16: result date 2021-10-10, layout archive date 2021-10-03; geometry unused.
- 2024 R22: result date 2024-11-23, layout archive date 2024-11-24; geometry unused.

## Feature importance

Global LightGBM gain from the final held-out fold only, trained before that race. These are predictive proxies, not causal driver/car decompositions or per-driver SHAP.

### Global

| Feature | Gain share |
|---|---:|
| driver_points_mean_10 | 13.3% |
| constructor_points_mean_10 | 12.5% |
| driver_quali_mean_10 | 11.4% |
| driver_points_mean_5 | 9.9% |
| constructor_finish_mean_10 | 5.2% |
| constructor_quali_mean_10 | 3.8% |
| driver_season_rank_pre | 3.0% |
| constructor_points_mean_5 | 2.9% |
| practice_fp2_position_pct | 2.0% |
| driver_points_mean_3 | 1.8% |
| driver_grid_mean_10 | 1.6% |
| rating_quali_combined | 1.3% |

### Ratings

| Feature | Gain share |
|---|---:|
| driver_quali_mean_10 | 19.3% |
| constructor_points_mean_10 | 13.0% |
| driver_points_mean_10 | 11.2% |
| driver_points_mean_5 | 5.3% |
| constructor_finish_mean_10 | 4.2% |
| constructor_points_mean_5 | 2.8% |
| constructor_quali_mean_10 | 2.7% |
| practice_fp2_position_pct | 2.5% |
| driver_season_rank_pre | 2.3% |
| driver_points_mean_3 | 1.9% |
| driver_quali_mean_5 | 1.3% |
| practice_fp1_gap_pct | 1.2% |

## Decision and limitations

Keep production frozen at this checkpoint. A development-profile retrospective comparison is not proof of full deployment improvement. Driver and constructor effects are not uniquely identifiable from observational results; strategy, team orders, traffic and status-label noise remain confounders. Classification/status labels are coarse; these ratings are not fuel-corrected telemetry pace estimates.

2026 remains reused retrospective validation. The test roster is the archived entrant set; source publication/revision timing is not reconstructed. Monte Carlo convergence across seeds and higher simulation counts is not established. Bootstrap intervals are approximate and not corrected for repeated model selection. Only pre-quali is evaluated here; full-profile pre/post-quali promotion checks belong to stage 5.

Artifacts: `metrics.json`, `predictions.parquet`, `race_scores.csv`, `calibration.csv`, `calibration.png`, `rating_snapshots.parquet`, `feature_manifest.json`, `audit.json`, `cutoffs.csv` and `importance_*.csv`.

Sources: [F1DB](https://github.com/f1db/f1db), [official team lineage history](https://www.formula1.com/en/latest/article/the-family-tree-f1-11-teams-and-how-they-came-to-be.2QBA1PPMf0bC8mp2xxqZeq.2QBA1PPMf0bC8mp2xxqZeq), [regulation-era history](https://www.formula1.com/en/latest/article/the-key-regulation-changes-in-f1-history-and-the-teams-that-nailed-them.2iq8c5E6S1HOffT5PBLl6i), and [2026 rules](https://www.formula1.com/en/page/2026-f1-regulations).
