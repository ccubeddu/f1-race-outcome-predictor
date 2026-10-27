# Historical data expansion — stage 1

Status: passed_with_coverage_caveats. Production unchanged: yes (SHA-256 verified).

Completed races: **191 → 347** (+156). Driver-race rows: **3854 → 7307**.

This is a staged data expansion. Model training, features, predictions and evaluation artifacts in the production data directory were not replaced.

## Season coverage

| Season | Races | Driver rows | Qualifying match | Practice rows | Timed laps | Pit-stop rows |
|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 19 | 456 | 100.0% | 1363 | 97.4% | 0 |
| 2011 | 19 | 454 | 99.1% | 1359 | 98.7% | 1114 |
| 2012 | 20 | 478 | 99.2% | 1430 | 98.3% | 946 |
| 2013 | 19 | 418 | 100.0% | 1252 | 98.6% | 951 |
| 2014 | 19 | 407 | 99.5% | 1214 | 97.7% | 816 |
| 2015 | 19 | 378 | 98.9% | 1100 | 98.0% | 741 |
| 2016 | 21 | 462 | 98.9% | 1381 | 98.3% | 989 |
| 2017 | 20 | 400 | 99.5% | 1173 | 96.5% | 694 |
| 2018 | 21 | 420 | 100.0% | 1254 | 97.9% | 536 |
| 2019 | 21 | 420 | 99.5% | 1236 | 97.6% | 650 |
| 2020 | 17 | 340 | 100.0% | 918 | 97.9% | 594 |
| 2021 | 22 | 440 | 99.8% | 1238 | 99.7% | 798 |
| 2022 | 22 | 440 | 100.0% | 1254 | 98.3% | 805 |
| 2023 | 22 | 440 | 100.0% | 1073 | 97.6% | 912 |
| 2024 | 24 | 479 | 100.0% | 1190 | 98.0% | 825 |
| 2025 | 24 | 479 | 99.8% | 1197 | 99.6% | 821 |
| 2026 | 18 | 396 | 98.7% | 920 | 99.1% | 721 |


## Coverage caveats before feature promotion

Older practice classifications: 10,272 rows; timed laps 97.9%. Practice-source date coverage 0.0%; time coverage 0.0%.

Timing evidence from the independent Jolpica schedule: {"before_qualifying_calendar_day": 310, "same_day_without_exact_times": 156}. Scheduled times do not prove actual completion/publication. Missing dates and same-day sessions without times need a conservative feature policy.

Detected 149 practice rows where the same car number belongs to a different driver in the race results. A car-number-only join can attribute a substitute's practice to the race driver; this must be fixed in the next feature stage.

Jolpica pit-stop data starts in 2011; 2010 is recorded as unavailable, not zero pit stops. Detailed tyre/lap telemetry does not become available simply by extending the classification history.

Driver-entry counts differ between the sources in 2 races. Per-race differences are retained in audit.json for reconciliation.

Source-only DNQ/DNP entries identified: 4. These are kept as source-audit evidence rather than assigned invented finishing positions.

Sources: [Jolpica](https://github.com/jolpica/jolpica-f1), [F1DB, CC-BY-4.0](https://github.com/f1db/f1db). F1DB archive SHA-256: `b93aef12ac5b3a6c05d0ee3c78b1822033ff7da531fb19b6ddad79c6361a2bff`.

## Next stage

Audit conservative practice eligibility, identity joins, normalized historical features and recency weighting in the staging dataset. Compare 2018+, 2014+ and 2010+ using identical chronological evaluation races and production model settings. No accuracy gain is claimed at this data-only stage.
