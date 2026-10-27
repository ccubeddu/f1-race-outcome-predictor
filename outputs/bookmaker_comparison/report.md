# Model vs bookmakers

Winner odds are converted to implied probabilities and normalized across the complete 22-driver
market. This removes the bookmaker overround with the standard multiplicative de-vig method.

## Audit correction — previous edge claim withdrawn

The previous 4/4 model versus 2/4 bookmaker headline compares different, unverified information
cutoffs. A search index's crawl date does not establish when the bookmaker offered a price.
The model includes qualifying and final starting-grid features; the bookmaker boards could predate
those inputs. Consequently these four races are excluded from the timing-eligible scoreboard.
Their arithmetic remains below as exploratory analysis for reproducibility.

The Austin pre-quali checkpoint has a similar mismatch: October 19 odds predate practice, while
the stored pre-quali model uses same-weekend practice. The direction of that particular score does
not make the comparison fair. Neither historical result establishes which forecaster is better.

The default CLI now reports **0 timing-eligible races** for the existing bookmaker manifest.
Both model and odds need exact timestamps and a documented common information cutoff.
Also, 2026 has been reused during model selection, so the model backtests are retrospective
validation rather than an untouched confirmatory holdout.

## Original post-quali arithmetic — timing unverified

The model forecasts are strict walk-forward predictions: every race was predicted by a model
trained only on earlier races. The bookmaker side uses complete 22-driver BetMGM race-winner boards
preserved by search indexing. Those pages do not expose an exact price-capture timestamp.

| Race | Winner | Model favorite | Market favorite | Model P(winner) | Market P(winner) | Model LL | Market LL |
|---|---|---|---|---:|---:|---:|---:|
| Spain | Kimi Antonelli | Kimi Antonelli | Kimi Antonelli | 45.9% | 29.8% | **0.780** | 1.212 |
| Bahrain in Malaysia | Max Verstappen | Max Verstappen | Kimi Antonelli | 80.7% | 10.3% | **0.214** | 2.269 |
| Singapore | Max Verstappen | Max Verstappen | Max Verstappen | 41.5% | 25.0% | **0.879** | 1.386 |
| Austin | Kimi Antonelli | Kimi Antonelli | Max Verstappen | 67.3% | 24.4% | **0.396** | 1.410 |

| Forecast | Favorite accuracy | Mean log loss | Mean multiclass Brier |
|---|---:|---:|---:|
| Post-quali model | **4/4 (100%)** | **0.567** | **0.283** |
| De-vigged market | 2/4 (50%) | 1.569 | 0.734 |

Lower log loss and Brier scores are better. The arithmetic favors the model on all four races,
but the unknown information cutoff prevents attributing that difference to model quality.

Austin demonstrates sensitivity to the information set: the model's winner probability changed
after qualifying. Without a post-quali timestamp for the odds, that change cannot be called a win
over the bookmaker.

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

## Austin 2026 — illustrative arithmetic, cutoffs differ

This uses the leakage-safe pre-quali walk-forward prediction trained only through October 11 and
Kambi race-winner prices observed October 19. Andrea Kimi Antonelli won.

| Forecast | Favorite | P(actual winner) | Winner log loss | Multiclass Brier |
|---|---|---:|---:|---:|
| Model | Max Verstappen | 17.7% | 1.732 | 0.907 |
| De-vigged market | Max Verstappen | 22.9% | **1.475** | **0.794** |

Both forecasts chose the wrong favorite. The market assigned more probability to Antonelli and
scores better arithmetically, but the October 19 board and practice-enriched forecast have different
information cutoffs. Treat the result as illustrative only.

## Reproduce

```bash
f1predict odds compare \
  outputs/predictions/2026_r19_mexico-city-grand-prix_pre_quali.csv \
  outputs/bookmaker_comparison/mexico_2026_lucky_rebel.csv

f1predict odds compare \
  outputs/bookmaker_comparison/austin_2026_pre_quali_walk_forward.csv \
  outputs/bookmaker_comparison/austin_2026_kambi.csv \
  --winner antonelli

f1predict odds history \
  outputs/bookmaker_comparison/history_post_quali.csv \
  --output outputs/bookmaker_comparison/history_post_quali_scores.csv

# Explicitly opt into the original, unverified aggregate
f1predict odds history outputs/bookmaker_comparison/history_post_quali.csv --include-unverified
```

The input must contain every driver and exactly one of `decimal_odds`, `american_odds`, or
`fractional_odds`. Complete markets are required because partial boards cannot be de-vigged
correctly.

Forecast comparison only—not betting advice. Gamble responsibly.
