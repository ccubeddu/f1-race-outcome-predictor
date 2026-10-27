# Model vs bookmakers

Winner odds are converted to implied probabilities and normalized across the complete 22-driver
market. This removes the bookmaker overround with the standard multiplicative de-vig method.

## Mexico City 2026 — prospective pre-quali comparison

Snapshot: Lucky Rebel race-winner prices on October 26, 2026. The model was trained through the
October 25 United States Grand Prix. Neither forecast has seen Mexico practice or qualifying.

| Driver | Model P(win) | Market fair P(win) | Difference |
|---|---:|---:|---:|
| Max Verstappen | 36.7% | 35.3% | +1.4 pp |
| Kimi Antonelli | 26.0% | 31.3% | -5.3 pp |
| Lewis Hamilton | 10.3% | 6.4% | +3.9 pp |
| Lando Norris | 10.0% | 4.9% | +5.1 pp |
| Charles Leclerc | 8.6% | 7.5% | +1.1 pp |
| George Russell | 3.6% | 5.5% | -2.0 pp |
| Isack Hadjar | 3.1% | 3.9% | -0.8 pp |
| Oscar Piastri | 0.8% | 2.4% | -1.7 pp |

Both select Verstappen as the favorite and Antonelli second. The largest disagreements are Norris
(model +5.1 percentage points), Antonelli (-5.3), Hamilton (+3.9), and Russell (-2.0). The total
variation distance between the two full winner distributions is 11.5%. The bookmaker's raw
implied probabilities total 120.7%, a 20.7% overround before normalization.

This is a prospective snapshot, not an accuracy result. It should be scored only after the race.

## Austin 2026 — completed-race checkpoint

This uses the leakage-safe pre-quali walk-forward prediction trained only through October 11 and
Kambi race-winner prices observed October 19. Andrea Kimi Antonelli won.

| Forecast | Favorite | P(actual winner) | Winner log loss | Multiclass Brier |
|---|---|---:|---:|---:|
| Model | Max Verstappen | 17.7% | 1.732 | 0.907 |
| De-vigged market | Max Verstappen | 22.9% | **1.475** | **0.794** |

Both forecasts chose the wrong favorite, but the market assigned more probability to Antonelli and
therefore scored better on both proper scoring rules. One race is not enough to claim that either
forecast is generally superior. A meaningful comparison needs the same pre-quali or post-quali
snapshot for many races.

## Reproduce

```bash
f1predict odds compare \
  outputs/predictions/2026_r19_mexico-city-grand-prix_pre_quali.csv \
  outputs/bookmaker_comparison/mexico_2026_lucky_rebel.csv

f1predict odds compare \
  outputs/bookmaker_comparison/austin_2026_pre_quali_walk_forward.csv \
  outputs/bookmaker_comparison/austin_2026_kambi.csv \
  --winner antonelli
```

The input must contain every driver and exactly one of `decimal_odds`, `american_odds`, or
`fractional_odds`. Complete markets are required because partial boards cannot be de-vigged
correctly.

Forecast comparison only—not betting advice. Gamble responsibly.
