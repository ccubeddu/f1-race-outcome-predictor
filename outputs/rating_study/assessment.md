# Stage 4 assessment

Matched pre-qualifying development-profile comparisons on the expanded staged dataset. Production stays frozen; the old deployed model is not this refitted baseline.

| Candidate | Spearman ↑ | Position LL ↓ | Winner LL ↓ | DNF Brier ↓ |
|---|---:|---:|---:|---:|
| Direct baseline | 0.6226 | 2.7881 | 1.4278 | 0.11256 |
| Ratings + circuit | 0.6236 | 2.7542 | 1.4223 | 0.11256 |
| Ratings + circuit + reliability | 0.6222 | 2.7477 | 1.4340 | 0.11248 |

## What changed

Driver and constructor pairwise ratings have separate qualifying and classified-finisher channels. Teammate comparisons, retained driver history across moves, discounted team lineages, layout-specific residuals and coarse geometry buckets are tested explicitly. A separate DNF variant adds cause-typed reliability histories; it reuses the same finish models as the circuit variant. No rating reads current-race outcomes.

## Evidence

- Ratings + circuit: +1.22% position-loss reduction; winner-loss change -0.0055. Positive loss reduction is better.
- Ratings + circuit + reliability: +1.45% position-loss reduction; winner-loss change +0.0062. Positive loss reduction is better.
- Reliability minus baseline mean per-race DNF Brier: -0.00007, approximate paired block-bootstrap 95% interval [-0.00038, +0.00027]. Negative is better. The main table averages driver rows; this interval gives each race equal weight.

Full ranker-only feature ablations, per-season/per-era results, paired intervals, calibration plots and feature importance are in `report.md` and `metrics.json`. Do not choose a candidate solely by one metric or one season.

## Checkpoint decision

Retain ratings + circuit as a research candidate for stage 5, not a deployment winner. Inspect the paired rank/winner-loss intervals as well as exact-position loss. Do not add the separate reliability model to production solely on this experiment.

Top-three overlap: 56.1% baseline versus 55.7% with ratings + circuit. Top-ten overlap: 76.4% versus 75.9%.

The stage 3 qualifying-grid mixture remains a separate candidate. These experiments do not establish that stacking the two additions improves results.

## Limits and next gate

These ratings are not uniquely identifiable causal measures of driver skill or car quality. Source status labels are coarse; unknown retirements are not called mechanical failures, and incident labels do not prove driver fault. Race-rank ratings are not fuel/traffic-corrected telemetry pace.

Only 345/347 layout dates verify; geometry for Turkey 2021 and Las Vegas 2024 is excluded because dates disagree across sources. Complete pre-race snapshots and their prior-event dates are saved in `rating_snapshots.parquet`.

This is reused retrospective validation with a known archived entrant set, not a prospective forecast. Publication/revision timing and Monte Carlo convergence are not established. Stage 5 still needs matched full-production settings, training-window and recency tests, both pre/post-quali cutoffs and a clear promotion decision.

In the reused 2026 season, Spearman changes from 0.6545 to 0.6462 with ratings + circuit. Check this era separately rather than relying on the aggregate score.
