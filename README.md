# F1 Predict

A leakage-safe Formula 1 outcome predictor with pre-qualifying and post-qualifying modes.

The project provides reproducible data ingestion, explicit pre/post-qualifying feature matrices,
calibrated joint race simulation, walk-forward evaluation, and a simple prediction CLI.

## Austin example

For the 2026 United States Grand Prix, the post-qualifying walk-forward model was trained only
through October 11—before the Austin race—and correctly predicted Andrea Kimi Antonelli to win.
It identified all three podium drivers and all ten points finishers.

| Pred. | Driver | Expected | P(win) | P(podium) | P(points) | Actual |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Andrea Kimi Antonelli | 2.70 | 67.3% | 91.6% | 92.5% | 1 |
| 2 | Max Verstappen | 4.22 | 22.1% | 83.1% | 87.6% | 3 |
| 3 | Charles Leclerc | 4.96 | 4.5% | 50.5% | 91.0% | 2 |
| 4 | Isack Hadjar | 6.15 | 3.4% | 36.1% | 87.2% | 6 |
| 5 | Lewis Hamilton | 6.20 | 1.0% | 15.4% | 91.4% | 4 |
| 6 | Lando Norris | 7.31 | 0.8% | 10.8% | 88.0% | 5 |
| 7 | George Russell | 7.52 | 0.6% | 7.3% | 86.4% | 8 |
| 8 | Oscar Piastri | 8.73 | 0.2% | 2.5% | 82.6% | 7 |
| 9 | Liam Lawson | 11.36 | 0.0% | 0.6% | 55.6% | 9 |
| 10 | Pierre Gasly | 12.31 | 0.0% | 0.3% | 39.5% | 10 |

Race metrics: **0.916 Spearman**, **1.73-position MAE**, and **2.136 position log loss**.
See the [full 22-driver forecast](outputs/examples/austin_2026/report.md),
[CSV](outputs/examples/austin_2026/post_quali_walk_forward.csv), or
[JSON](outputs/examples/austin_2026/post_quali_walk_forward.json).

## Quick start

```bash
make setup
source .venv/bin/activate
f1predict doctor
f1predict data sync --start 2018 --end 2025
f1predict data audit
f1predict data sync-practice --start 2018 --end 2025
f1predict features build
f1predict features audit
f1predict model train --mode all
f1predict predict 19 --season 2026
```

The default sync downloads compact Jolpica data. Add FastF1 session detail when you want the
lap, tyre, and weather layer:

```bash
f1predict data sync-fastf1 --start 2018 --end 2025 --sessions FP1,FP2,FP3,Q,R
```

To fetch or retry only selected events, add `--rounds 1,2`.

FastF1's first download is intentionally separate because several seasons of session timing
data can take a long time. Both sources are cached, so later runs are incremental.

As of October 2026, Formula 1 requires authentication for its historical timing archive.
FastF1 therefore needs an F1 TV Access (or higher) subscription for lap and weather downloads.
Set `F1_USERNAME` and `F1_PASSWORD` in your shell before `sync-fastf1`; credentials are read by
FastF1 and are never written by this project. Jolpica classifications remain available without
an account, and the predictor will support a Jolpica-only feature profile.

The default practice-data path does not require an F1 account:

- F1DB provides FP1/FP2/FP3 classifications, best laps, and lap counts for 2018 onward.
- OpenF1 adds every practice lap, sector splits, speed traps, tyre stints, and weather from 2023.
- FastF1 remains an optional enrichment source.

Both sources are downloaded with one resumable command:

```bash
f1predict data sync-practice --start 2018 --end 2026
```

OpenF1's public API is rate limited. A full detailed backfill may take multiple runs; completed
session partitions are skipped automatically. Use `--rounds 1,2` when only selected events are
needed. F1DB classification coverage is downloaded in one compact release and is not affected by
the OpenF1 rate limit.

## Data layout

```text
data/
  raw/jolpica/          # exact API responses, content-addressed by request
  raw/f1db/             # versioned CC-BY-4.0 CSV release
  raw/openf1/           # exact historical API responses
  cache/fastf1/         # FastF1's own HTTP cache
  processed/jolpica/    # partitioned normalized tables
  processed/f1db/       # historical practice classifications
  processed/openf1/     # detailed practice laps, stints, and weather
  processed/fastf1/     # partitioned session tables
  features/             # pre_quali.parquet, post_quali.parquet, and manifest.json
  catalog.json          # refresh metadata and row counts
```

Generated datasets are excluded from Git. The raw cache preserves reproducibility without
mixing source payloads into model-ready tables.

## Feature layer and prediction cutoffs

Build both model-ready datasets with:

```bash
f1predict features build
f1predict features audit
```

The builder creates one row per race and driver in two explicit modes:

- `pre_quali`: prior-race driver and constructor form, track history, teammate head-to-head,
  reliability, pit-stop history, championship context, and same-weekend practice information
  scheduled before qualifying.
- `post_quali`: every pre-qualifying feature plus the current qualifying classification, Q1/Q2/Q3
  gaps, grid position, advancement flags, and the current teammate qualifying delta.

Historical rolling and expanding statistics always use `shift(1)`. Race outcomes are labels
only. Practice eligibility is checked against the published event schedule; this excludes FP2
held after qualifying on early Sprint weekends. The generated `manifest.json` records every
feature, its prediction cutoff, and the leakage policy used to build it.

Current historical build (2018–2026 results available locally):

| Mode | Race-driver rows | Races | Features |
|---|---:|---:|---:|
| Pre-qualifying | 3,854 | 191 | 138 |
| Post-qualifying | 3,854 | 191 | 148 |

## Model and simulation

Train both prediction cutoffs with:

```bash
f1predict model train --mode all
f1predict model inspect --mode post_quali
f1predict model importance --mode post_quali --top 20
```

Each artifact contains:

- An anchored ensemble of LightGBM LambdaRank, Random Forest, and LightGBM regression members.
- Mode-specific ensemble weights verified on a later-season holdout. Post-quali uses 60%
  LambdaRank, 30% Random Forest, and 10% LightGBM regression.
- A separate LightGBM DNF classifier with Platt calibration on the chronologically latest
  30-race calibration window.
- Per-member race temperatures calibrated from winner log loss on that same chronological window.
- Its exact feature schema, data cutoff, random seed, package versions, and calibration metadata.

The evaluation stage compares the ranker with a random-forest supervised baseline and a
LightGBM finish-position regressor. The production ensemble combines their finishing-position
distributions, then uses iterative proportional fitting within the win, podium, points, and
non-points regions. This improves exact finishing order while preserving LambdaRank's calibrated
P(win), P(podium), and P(points) exactly.

Post-qualifying predictions then apply a 1.25× pole-odds adjustment selected on the 2024–2025
development period and confirmed on the 2026 holdout. The position matrix is rebalanced afterward,
so grid win probabilities still sum to 100% and every position still has exactly one driver.
P(podium) and P(points) remain unchanged for every driver.

Race probabilities come from joint Monte Carlo simulations, not independent per-driver
classifiers. Every simulated race assigns one driver to every finishing position. This guarantees
that grid win probabilities sum to 100%, every driver's position distribution sums to 100%, and
the expected finishing order has no duplicate positions. Position columns expand to the actual
grid size, including the 22-driver 2026 grid, while points remain defined as the top 10.

On macOS, LightGBM also requires the OpenMP runtime:

```bash
brew install libomp
```

## Walk-forward evaluation

Run the expanding-window backtest and inspect its integrity checks with:

```bash
f1predict evaluate run --mode all --start-season 2024 --simulations 3000
f1predict evaluate report
f1predict evaluate audit
```

For every test race, all candidate models are rebuilt using only races that happened earlier.
Probability temperature and DNF calibration are updated only from earlier out-of-fold predictions.
The stored prediction table includes the last training date for an explicit chronology audit.

Results over 66 completed races from 2024 through 2026:

| Mode | Model | Spearman | Top 3 | Top 10 | Position MAE | Position log loss |
|---|---|---:|---:|---:|---:|---:|
| Pre-quali | **Anchored ensemble** | 0.632 | **56.6%** | 75.6% | **3.65** | **2.793** |
| Pre-quali | Random Forest | 0.633 | 52.0% | 75.9% | 3.66 | 2.896 |
| Pre-quali | LightGBM regression | 0.632 | 50.0% | 77.3% | 3.67 | 3.511 |
| Pre-quali | LightGBM LambdaRank | 0.626 | 56.1% | 75.6% | 3.70 | 2.893 |
| Post-quali | **Pole-aware ensemble** | **0.694** | **68.2%** | 78.9% | **3.21** | **2.587** |
| Post-quali | **Anchored ensemble** | **0.694** | **68.2%** | 78.9% | **3.21** | **2.589** |
| Post-quali | Random Forest | 0.691 | 60.6% | 78.6% | 3.27 | 2.639 |
| Post-quali | LightGBM regression | 0.688 | 61.1% | **79.1%** | 3.31 | 3.107 |
| Post-quali | LightGBM LambdaRank | 0.686 | 66.2% | 78.6% | 3.26 | 2.669 |

The 2026-only holdout confirms that the improvements are not confined to the weight-development
period. The final post-quali mixture scores 70.4% top-three accuracy and 2.633 position log loss
on the 18-race 2026 holdout. Pre-quali also improves Spearman, top-three accuracy, MAE, and
position log loss on the holdout relative to its single-model baseline.

The post-quali pole adjustment retains 60.6% highest-P(win) accuracy and 40.5% win Brier skill
over the full window. The three-model mixture improves top-three accuracy by one percentage point
and reduces position log loss by about 1% versus the previous two-model production ensemble.

DNF probability is calibrated but currently has little Brier skill over a base-rate forecast.
A conservative shrinkage experiment did not improve both the development and holdout periods, so
it was rejected rather than being promoted on an unstable gain. Richer component-level reliability
data remains the clearest future improvement.

The evaluation directory contains driver-level predictions, JSON metrics, calibration tables,
calibration plots, a model-comparison chart, and feature-importance graphics.

## Compare with bookmaker odds

Use a complete race-winner market CSV with `driver_id` plus exactly one odds column:
`decimal_odds`, `american_odds`, or `fractional_odds`. The command removes the bookmaker margin,
joins every driver to the model forecast, and reports probability differences and fair prices.

```bash
f1predict odds compare \
  outputs/predictions/2026_r19_mexico-city-grand-prix_pre_quali.csv \
  outputs/bookmaker_comparison/mexico_2026_lucky_rebel.csv
```

After the race, add the actual winner to score both full winner distributions with log loss and
multiclass Brier score:

```bash
f1predict odds compare model_prediction.csv bookmaker_odds.csv --winner antonelli
```

For a multi-race scoreboard, provide a manifest with `race`, `prediction_csv`, `odds_csv`, and
`winner` columns:

```bash
f1predict odds history outputs/bookmaker_comparison/history_post_quali.csv
```

The included four-race post-quali sample currently has 100% model favorite accuracy versus 50% for
the bookmaker favorites, with mean winner log loss of 0.567 versus 1.569. This is promising but
still a small sample; the archived BetMGM pages also lack exact capture timestamps, so it is not
presented as a definitive closing-line benchmark.

The repository includes a [Mexico prospective comparison](outputs/bookmaker_comparison/report.md)
and an Austin completed-race checkpoint. Historical odds are not freely available as one clean
dataset, so broad bookmaker claims are deliberately deferred until more same-cutoff snapshots have
been collected. Forecast comparison only—not betting advice.

## Predict a race

The default `auto` mode uses the post-qualifying model when a qualifying classification is
available and falls back to pre-qualifying otherwise:

```bash
# Full grid; round number or a race/circuit/location name both work
f1predict predict 19 --season 2026
f1predict predict monza --season 2026

# Ask about one driver
f1predict predict monza --season 2026 --driver verstappen

# Historical outcomes for a driver from one qualifying position
f1predict stats driver-quali verstappen 2

# Force a cutoff or return machine-readable output
f1predict predict 19 --season 2026 --mode pre_quali --json
```

Each forecast shows expected finishing order, P(win), P(podium), P(points), and P(DNF). A
single-driver query also shows that driver's complete probability distribution over every grid
position. By default, the same complete result is saved as CSV and JSON under
`outputs/predictions/`; use `--no-save` for terminal-only queries.

For a future event, run the data sync immediately before predicting. The live roster comes from
the target event's qualifying classification in post-quali mode, then pre-quali practice in
pre-quali mode, with the latest completed race as a fallback. Post-quali currently treats the
qualifying order as the provisional starting grid, so grid penalties announced later are not yet
applied automatically. Historical queries are convenient replays using the current fully trained
model; the walk-forward artifacts—not replay output—are the unbiased measure of historical
performance.

## Useful commands

```bash
f1predict --help
f1predict data sync --help
f1predict data sync-fastf1 --help
f1predict data audit --json
f1predict features build
f1predict features audit --json
f1predict model train --mode all
f1predict model importance --mode pre_quali --top 15
f1predict evaluate run --mode all --start-season 2024
f1predict evaluate audit
f1predict odds compare model_prediction.csv bookmaker_odds.csv
f1predict odds history history_manifest.csv
f1predict predict monza --season 2026 --driver verstappen
f1predict stats driver-quali verstappen 2
f1predict doctor
```

## Development

```bash
make test
make lint
```

The automated suite covers source normalization, leakage-safe rolling features, live unlabelled
race construction, coherent probability simulation, training, evaluation, and driver lookup.
