# Practice feature checkpoint — stage 2

Built 7,307 driver-race rows over 347 races; 148 pre-quali / 158 post-quali features.

Production fingerprints match the frozen baseline. No model trained or promoted; no improvement in prediction metrics is claimed.

## Coverage

Percentages are driver-race availability, not percentages of all raw practice sessions. Legacy detail includes the old eligibility rules; staged detail uses the corrected rules.

| Season | Races | Legacy detail | Staged detail | Tyre-aware long run |
|---|---:|---:|---:|---:|
| 2010 | 19 | — | 0.0% | 0.0% |
| 2011 | 19 | — | 0.0% | 0.0% |
| 2012 | 20 | — | 0.0% | 0.0% |
| 2013 | 19 | — | 0.0% | 0.0% |
| 2014 | 19 | — | 0.0% | 0.0% |
| 2015 | 19 | — | 0.0% | 0.0% |
| 2016 | 21 | — | 0.0% | 0.0% |
| 2017 | 20 | — | 0.0% | 0.0% |
| 2018 | 21 | 0.0% | 0.0% | 0.0% |
| 2019 | 21 | 0.0% | 0.0% | 0.0% |
| 2020 | 17 | 0.0% | 0.0% | 0.0% |
| 2021 | 22 | 0.0% | 0.0% | 0.0% |
| 2022 | 22 | 0.0% | 0.0% | 0.0% |
| 2023 | 22 | 50.0% | 99.1% | 84.3% |
| 2024 | 24 | 4.2% | 100.0% | 89.8% |
| 2025 | 24 | 4.2% | 100.0% | 88.5% |
| 2026 | 18 | 5.6% | 99.7% | 85.6% |

## Leakage and identity checks

- 14,922/20,552 basic practice rows pass timing; 429 additionally excluded for unmatched driver identity.
- 144 eligible practice rows share an entrant's number but not their identity; none are attributed by number.
- Laps, stints and weather share the same completed-session cutoff, with additional per-record timestamp checks.
- Duplicate identities fail explicitly; missing detail and peer comparisons stay missing.
- Legacy model/input recipe mismatches fail explicitly, preserving old historical replay.

## Detailed source counts

| Table | Processed rows | Eligible rows | Excluded rows | Sessions |
|---|---:|---:|---:|---:|
| practice_laps | 99,349 | 97,433 | 1,916 | 216 |
| practice_stints | 18,274 | 17,966 | 308 | 216 |
| practice_weather | 18,037 | 17,623 | 414 | 216 |

## Limitations

- Coverage is not predictive accuracy. No model was retrained or promoted.
- Before/after coverage changes both acquisition and eligibility/identity rules.
- Event-time cutoffs do not reconstruct historical publication/revision timestamps.
- Without end timestamps, same-day legacy practice is excluded, including FP3.
- Long-run pace is a compound/age/session-relative proxy; fuel, traffic and flags are not controlled.
- A live pre-quali roster defaults provisionally to the previous race; practice is not an entry list.
- 2026 remains reused retrospective validation, not an untouched holdout.

## Next approval checkpoint

Proceed to chronological qualifying-grid modeling and paired feature ablations. Keep this feature recipe and old production model separate until backtests justify promotion.

Sources: [OpenF1 API documentation](https://openf1.org/docs/) and [free-tier access limits](https://openf1.org/). Bounded season queries are cached and partitioned into practice sessions only; requests are paced at least 2.1 seconds apart.
