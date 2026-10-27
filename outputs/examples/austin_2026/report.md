# Austin 2026 model example

This is the saved post-qualifying walk-forward prediction for the 2026 United States Grand Prix at Circuit of the Americas. All models were trained only through 2026-10-11, before the Austin race on 2026-10-25.

## Race-level result

- Winner prediction: correct (Andrea Kimi Antonelli).
- Podium-driver overlap: 100%.
- Points-finisher overlap: 100%.
- Spearman rank correlation: 0.916.
- Finishing-position MAE: 1.73 places.
- Exact-position log loss: 2.136.

## Forecast versus result

| Pred. | Driver | Team | Expected | Win | Podium | Points | DNF | Actual |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Andrea Kimi Antonelli | Mercedes | 2.70 | 67.3% | 91.6% | 92.5% | 7.7% | 1 |
| 2 | Max Verstappen | Red Bull | 4.22 | 22.1% | 83.1% | 87.6% | 12.6% | 3 |
| 3 | Charles Leclerc | Ferrari | 4.96 | 4.5% | 50.5% | 91.0% | 8.9% | 2 |
| 4 | Isack Hadjar | Red Bull | 6.15 | 3.4% | 36.1% | 87.2% | 12.2% | 6 |
| 5 | Lewis Hamilton | Ferrari | 6.20 | 1.0% | 15.4% | 91.4% | 8.8% | 4 |
| 6 | Lando Norris | McLaren | 7.31 | 0.8% | 10.8% | 88.0% | 12.5% | 5 |
| 7 | George Russell | Mercedes | 7.52 | 0.6% | 7.3% | 86.4% | 13.5% | 8 |
| 8 | Oscar Piastri | McLaren | 8.73 | 0.2% | 2.5% | 82.6% | 11.4% | 7 |
| 9 | Liam Lawson | RB F1 Team | 11.36 | 0.0% | 0.6% | 55.6% | 15.3% | 9 |
| 10 | Pierre Gasly | Alpine F1 Team | 12.31 | 0.0% | 0.3% | 39.5% | 14.7% | 10 |
| 11 | Arvid Lindblad | RB F1 Team | 12.99 | 0.1% | 0.3% | 31.2% | 14.1% | 12 |
| 12 | Nico Hülkenberg | Audi | 13.14 | 0.0% | 0.3% | 28.1% | 13.1% | 11 |
| 13 | Gabriel Bortoleto | Audi | 13.48 | 0.0% | 0.4% | 23.9% | 12.8% | 13 |
| 14 | Franco Colapinto | Alpine F1 Team | 14.53 | 0.0% | 0.1% | 16.8% | 13.7% | 21 |
| 15 | Oliver Bearman | Haas F1 Team | 14.56 | 0.0% | 0.1% | 15.8% | 14.2% | 22 |
| 16 | Esteban Ocon | Haas F1 Team | 14.96 | 0.0% | 0.0% | 15.2% | 13.6% | 14 |
| 17 | Fernando Alonso | Aston Martin | 15.21 | 0.0% | 0.1% | 13.2% | 15.7% | 16 |
| 18 | Carlos Sainz | Williams | 16.18 | 0.0% | 0.1% | 11.7% | 16.5% | 18 |
| 19 | Lance Stroll | Aston Martin | 16.34 | 0.0% | 0.1% | 11.1% | 19.2% | 15 |
| 20 | Alexander Albon | Williams | 16.51 | 0.0% | 0.0% | 10.6% | 17.7% | 17 |
| 21 | Sergio Pérez | Cadillac F1 Team | 16.79 | 0.0% | 0.1% | 10.3% | 21.9% | 19 |
| 22 | Valtteri Bottas | Cadillac F1 Team | 16.87 | 0.0% | 0.0% | 10.4% | 22.9% | 20 |

The CSV and JSON contain the complete probability distribution over positions 1–22 for every driver. Actual finishing positions are included only for retrospective comparison; they were not available to the model at prediction time.
