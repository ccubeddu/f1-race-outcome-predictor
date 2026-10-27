# Approved improvement sequence

## 1. Expand historical coverage — current data checkpoint

Backfill 2010–2017 results, qualifying, pit stops where available, and practice classifications.
Keep a separate staging dataset, preserve the production artifacts, check duplicate/event keys,
compare race and driver counts with F1DB, and document timing and identity gaps. See `report.md`
and `audit.json` for the actual completed run. No improvement in model accuracy is claimed here.

## 2. Improve practice features and chronology

- Resolve driver identity before attributing a substitute's practice laps to a race driver. Keep
  car-level evidence separate from driver-level evidence.
- Apply one explicit information cutoff to laps, tyre stints and weather. Scheduled start times
  alone do not establish session completion or publication time.
- Exclude or explicitly flag legacy sessions whose pre-quali availability cannot be established.
- Fill missing detailed practice partitions from 2023 onward, respecting the source rate limits.
- Compare long-run pace by tyre compound, tyre age, session conditions and lap validity; distinguish
  qualifying-style runs from race simulations. Add availability indicators for sparse old data.
- Normalize relative performance and evaluate recency weighting and regulation-era context.

Checkpoint: coverage by season, leakage checks, identity-mismatch counts and feature ablations.

## 3. Model qualifying uncertainty before race outcomes

Fit a qualifying-order model using only features available before qualifying. Generate coherent
possible grids, feed them to a race model that supports those simulated inputs, and marginalize
over grids. Any predicted qualifying features used during race-model training must be generated
chronologically out of fold. Do not synthesize measured Q1/Q2/Q3 timing as if it were observed.

Checkpoint: qualifying rank/log-loss metrics and pre-quali race-probability improvements against
the existing direct model on identical chronological races.

## 4. Separate driver, constructor and circuit effects

Test an updating driver/constructor rating model with teammate comparisons and track-layout
context. Update ratings only after each race. Treat constructor lineage and driver moves
explicitly, and estimate changes in reliability separately from underlying pace where labels allow.

Checkpoint: feature/rating ablations, rank and probability scores, and performance by regulation era.

## 5. Evaluate the deployment configuration consistently

Use the same model configuration for backtesting and production. Compare 2018+, 2014+ and 2010+
training histories on identical test races, including variants with recency weighting. Tune on
earlier chronological windows; classify reused 2026 races as retrospective validation. Preserve
future frozen predictions for an untouched prospective comparison.

Report Spearman, position MAE, top-three/top-ten overlap, winner and position log loss, categorical
and binary Brier scores, DNF calibration and calibration plots. Use paired uncertainty estimates.
Keep sportsbook, exchange and prediction-market results separate and require documented common
information cutoffs before reporting a timing-eligible market advantage.

Checkpoint: a baseline-versus-candidate report with a clear promotion or rejection decision.
