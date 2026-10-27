# Better historical market benchmarks

We can now retrieve timestamped F1 prediction-market histories without an API key. Seven snapshots
were recovered across the four most recent completed races, with one additional snapshot rejected
for stale prices. The previous untimestamped bookmaker advantage claim has been withdrawn.

## Which data source to use

| Source | What is available | Practical fit |
|---|---|---|
| Polymarket | Public event metadata and token price histories; actual recent F1 data retrieved and archived here | Best immediately available comparison; label as a prediction market |
| Betfair Historical Data | Timestamped exchange streams, BASIC one-minute last-traded prices, higher-detail paid tiers | Preferred exchange archive path; requires an eligible account and selecting/downloading archive packages |
| Search-indexed sportsbook pages | Prices with uncertain historical capture times | Exploratory only; exclude from timing-eligible results |

Polymarket documents [public discovery](https://docs.polymarket.com/market-data/discover-markets)
and [timestamped price history](https://docs.polymarket.com/api-reference/markets/get-prices-history).
Its data are a separate market benchmark, not a bookmaker's prices. Betfair documents the
[free BASIC tier and archive intervals](https://www.betfair.com.au/hub/education/how-to-model/historical-data-sources/)
and its [account/download workflow](https://support.developer.betfair.com/hc/en-us/articles/12859956891932-How-Can-I-Make-HTTP-Requests-to-the-Historical-Data-API).
No real Betfair F1 archive has yet been downloaded or used to produce scores in this repository.

The Odds API's [published sports list](https://the-odds-api.com/sports-odds-data/sports-apis)
does not currently list F1. A historical endpoint alone is insufficient to recommend a provider;
F1 coverage, race-winner markets, and historical capture times must all be established.

## Recent results — retrospective analysis

These use the stored walk-forward model predictions and Polymarket observations before two fixed
cutoffs: 15 minutes before scheduled qualifying for pre-quali, and two hours before scheduled race
start for post-quali. Every included driver has a recorded price no more than five minutes old.
Probabilities are normalized over the supplied actual entrant grid.

| Mode | Races | Model correct favorite | Market correct favorite | Model log loss | Market log loss | Model Brier | Market Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pre-quali | 3 | 0/3 | 1/3 | 1.818 | **1.174** | 0.869 | **0.590** |
| Post-quali | 4 | 4/4 | 3/4 | **0.454** | 0.630 | **0.190** | 0.307 |

Lower log loss and Brier are better. The pre-quali market forecast scores better in all three
included races. Post-quali, the model scores better in Azerbaijan, Bahrain/Malaysia, and Austin;
the market scores better in Singapore. This points to pre-quali probabilities as a priority for
model improvement.

| Race | Actual winner | Model post-quali P(winner) | Market P(winner) | Better log loss |
|---|---|---:|---:|---|
| Azerbaijan, September 26 | George Russell | 72.0% | 63.8% | Model |
| Bahrain/Malaysia, October 4 | Max Verstappen | 80.7% | 56.9% | Model |
| Singapore, October 11 | Max Verstappen | 41.5% | 62.1% | Market |
| Austin, October 25 | Kimi Antonelli | 67.3% | 35.7% | Model |

Post-quali mean log loss is 27.9% lower for the model in this sample. The paired bootstrap 95%
interval for model-minus-market mean log loss is **[-0.505, +0.214]**, which spans zero; four races
cannot establish a stable advantage. This interval describes the observed sample and does not
correct for repeated model selection.

## Evidence limits

- The market observations have exact timestamps, but publication times for all historic model
  features have not been reconstructed. In particular, the stored post-quali features use final
  starting grids. Therefore the strict CLI still reports **zero fully timing-eligible comparisons**
  for these manifests; the numbers above remain retrospective research.
- The model configuration was refined using the 2026 data. Those races are reused validation,
  not an untouched holdout. The stored backtests also use a smaller training profile than the full
  production fits. Future predictions from a frozen configuration are needed for a confirmatory test.
- Every event includes a reserve-driver outcome and six Other/placeholder outcomes with no usable
  history. Missing history remains unknown. The reserve's observed excluded mass reaches **2.7%**
  in Bahrain pre-quali. The comparison is conditional on the mapped actual entrants; it is not a
  claim that the excluded markets had zero probability.
- Austin pre-quali failed the fixed freshness rule: Max's last observation was **578 seconds**
  old against the 300-second limit. The raw response and rejection are retained.
- Historical CLOB prices and Betfair BASIC last-traded prices do not establish executable bid/ask
  quotes, spreads, available size, commission, or achievable betting returns.

## Use the new data paths

Download an exact public snapshot using the included verified Austin market mapping:

```bash
f1predict odds fetch-polymarket \
  f1-united-states-grand-prix-winner-2026-10-25 \
  outputs/public_market_benchmark/post_quali/united-states/mapping.json \
  --prediction-csv outputs/public_market_benchmark/post_quali/united-states/prediction.csv \
  --cutoff 2026-10-25T18:00:00Z \
  --output outputs/austin_polymarket_snapshot.csv
```

This saves quotes, raw responses, source IDs, timestamps, unmatched outcomes, and hashes.
The default five-minute age limit is fixed before scoring.

For Betfair, extract an archive's single `.bz2` or NDJSON stream. Supply an explicit JSON map of
selection IDs to driver IDs and copy the market ID and exact winner-type code from its definition:

```bash
f1predict odds import-betfair downloaded-market.bz2 selection_map.json \
  --market-id SOURCE_MARKET_ID --market-type SOURCE_WINNER_TYPE \
  --cutoff ISO_UTC_CUTOFF --output outputs/betfair_snapshot.csv
```

The importer reconstructs deltas before the requested time and rejects in-play, suspended,
incomplete, stale, or wrongly mapped markets. The archive's race/event identity and type still
need verification by the collector; no account login or paid purchase is performed by this tool.

Replay all included public-market calculations entirely offline:

```bash
python scripts/export_public_market_benchmark.py
f1predict odds history outputs/public_market_benchmark/post_quali_manifest.csv
f1predict odds history outputs/public_market_benchmark/post_quali_manifest.csv --include-unverified
```

Each race directory includes its explicit mapping, raw response, quotes, model prediction,
comparison and metadata. `metrics.json` contains aggregate scores, uncertainty, and exclusions.

The next confirmatory test should preserve the code/model configuration, feature-availability
cutoff and raw prices together for each upcoming pre-quali and post-quali prediction, then score
after the official result using the same winner-settlement definition.
