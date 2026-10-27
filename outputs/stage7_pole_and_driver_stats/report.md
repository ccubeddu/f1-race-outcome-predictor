# Pole-aware winner calibration and driver qualifying-position tracker

## Production change

Post-qualifying forecasts now multiply the pole sitter's simulated win odds by 1.25 and renormalize the entire grid. The factor was selected on 2024–2025 and checked on the separate 2026 holdout. Iterative proportional fitting restores a coherent joint finishing matrix while preserving every driver's P(podium) and P(points).

## 66-race walk-forward comparison

| Metric | Anchored ensemble | Pole-aware | Change |
|---|---:|---:|---:|
| Predicted-P1 accuracy ↑ | 57.6% | 59.1% | +1.5 pp |
| Highest-P(win) accuracy ↑ | 59.1% | 60.6% | +1.5 pp |
| Winner categorical log loss ↓ | 1.165 | 1.151 | -0.014 |
| Win Brier ↓ | 0.02798 | 0.02759 | -0.00039 |
| Win Brier skill ↑ | 39.7% | 40.5% | +0.8 pp |
| Spearman ↑ | 0.694 | 0.695 | +0.000 |
| Top-3 accuracy ↑ | 67.7% | 67.2% | -0.5 pp |
| Top-10 accuracy ↑ | 78.9% | 78.9% | +0.0 pp |
| Position log loss ↓ | 2.614 | 2.612 | -0.001 |

Top-three set accuracy loses one correct pick across 66 races; P(podium) itself is unchanged by construction. All other displayed winner/probability metrics improve, and top-ten accuracy is unchanged.

## 2026 holdout

Winner categorical log loss improves from 0.823 to 0.786; win Brier improves from 0.01846 to 0.01776. Highest-P(win) accuracy remains 72.2%, so the probability improvement does not come from a less stable top pick.

## Driver-by-qualifying-position query

```bash
f1predict stats driver-quali verstappen 2
f1predict stats driver-quali leclerc 1 --start-season 2022 --json
```

Example: Max Verstappen has 35 races from P2, 12 wins (34.3%), 26 podiums (74.3%), 4 DNFs (11.4%), and a 2.39 average finish when classified.

The sparse driver/slot aggregates were also tested as direct model features. They were rejected because their small Spearman gain came with worse podium accuracy, MAE, and position log loss. The tracker remains available for analysis without degrading forecasts.
