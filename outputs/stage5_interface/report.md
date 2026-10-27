# Stage 5 — Production prediction interface

## Outcome

The predictor is runnable end to end from one CLI command. It supports a full-grid race forecast,
a single-driver query, automatic pre/post-qualifying model selection, terminal tables, JSON output,
and saved CSV/JSON artifacts.

The selected production model is **LightGBM LambdaRank**, paired with a separately calibrated
LightGBM DNF model and joint Monte Carlo race simulation. `auto` uses the post-qualifying model
when a qualifying classification is available, otherwise it uses the pre-qualifying model.

## Commands

```bash
# Full-grid forecast
f1predict predict 19 --season 2026

# Natural race lookup and single-driver query
f1predict predict monza --season 2026 --driver verstappen

# Explicit prediction cutoff
f1predict predict 19 --season 2026 --mode pre_quali
f1predict predict monza --season 2026 --mode post_quali

# Machine-readable output without writing files
f1predict predict 19 --season 2026 --json --no-save
```

The race argument accepts a round number or an unambiguous race, circuit, city, or country name.
Driver lookup accepts a name, code, or driver ID. Saved predictions go to `outputs/predictions/`.

## Output contract

Every driver receives:

- expected position and a unique predicted finishing rank;
- P(win), P(podium), P(points), and calibrated P(DNF);
- a full distribution over all finishing positions in the actual grid.

The simulation creates a complete ordering jointly for every race. Consequently:

- grid-wide win probabilities sum to 1;
- each driver's finishing-position probabilities sum to 1;
- every simulated race has exactly one driver in every position;
- output expands automatically beyond 20 positions (the current example has 22 drivers).

## Live race construction

For races not already in the historical feature matrix, the predictor builds an unlabelled target
race and passes it through the same leakage-safe feature pipeline as training. The roster source is
selected in this order:

1. target qualifying classification for post-quali predictions;
2. target pre-qualifying practice classification for pre-quali predictions;
3. the latest completed race as a fallback before an entry list is available.

Historical rolling and cumulative features exclude the target race using a one-race shift. Current
race outcomes are blank and never supplied as features. Post-quali currently treats qualifying
order as the provisional grid; penalties and later grid changes are not automatically applied.

## Production model decision

The 66-race walk-forward backtest favors LambdaRank for the user-facing objective. In post-quali
mode it produced 67.2% top-three accuracy and 3.26 expected-position MAE, outperforming the two
regression baselines on those measures. Random Forest had slightly better exact-position log loss,
but was materially weaker at identifying the podium. LambdaRank is therefore the default, with the
saved selection policy documenting Random Forest as the challenger.

## Final verification

- Ruff formatting and lint: passed.
- Pytest: 21 passed.
- Feature audit: passed for 3,854 rows across 191 races in both modes.
- Backtest chronology violations: 0.
- Backtest maximum probability-normalization error: 4.44e-16.
- Saved 22-driver production example: win sum 1.0, all 22 predicted positions unique, maximum
  per-driver position-distribution error 0.0.

## Known limitation

DNF calibration is operational, but its walk-forward Brier skill is only about 0.9% over the
base-rate forecast. Component-level reliability and incident data are the most valuable next data
upgrade. Historical CLI queries use the final fully trained model and are therefore demonstrations,
not unbiased backtests; use the Stage 4 evaluation artifacts for historical performance claims.
