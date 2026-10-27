# Austin 2026 model example

This is the saved post-qualifying walk-forward prediction for the 2026 United States Grand Prix at Circuit of the Americas. All models were trained only through 2026-10-11, before the Austin race on 2026-10-25.

## Race-level result

- Winner prediction: correct (Andrea Kimi Antonelli).
- Podium-driver overlap: 100%.
- Points-finisher overlap: 100%.
- Spearman rank correlation: 0.919.
- Finishing-position MAE: 1.64 places.
- Exact-position log loss: 2.160.

## Forecast versus result

| Pred. | Driver | Team | Expected | Win | Podium | Points | DNF | Actual |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Andrea Kimi Antonelli | Mercedes | 2.63 | 66.8% | 92.1% | 92.8% | 7.7% | 1 |
| 2 | Max Verstappen | Red Bull | 4.15 | 22.2% | 83.1% | 87.9% | 12.6% | 3 |
| 3 | Charles Leclerc | Ferrari | 4.88 | 4.7% | 50.8% | 91.7% | 8.9% | 2 |
| 4 | Lewis Hamilton | Ferrari | 6.18 | 1.0% | 15.4% | 91.8% | 8.8% | 4 |
| 5 | Isack Hadjar | Red Bull | 6.22 | 3.2% | 36.8% | 87.1% | 12.2% | 6 |
| 6 | Lando Norris | McLaren | 7.50 | 1.0% | 9.2% | 87.0% | 12.5% | 5 |
| 7 | George Russell | Mercedes | 7.52 | 0.6% | 6.8% | 86.5% | 13.5% | 8 |
| 8 | Oscar Piastri | McLaren | 8.80 | 0.2% | 2.8% | 82.0% | 11.4% | 7 |
| 9 | Liam Lawson | RB F1 Team | 11.18 | 0.0% | 0.7% | 56.8% | 15.3% | 9 |
| 10 | Pierre Gasly | Alpine F1 Team | 12.24 | 0.1% | 0.5% | 39.9% | 14.7% | 10 |
| 11 | Arvid Lindblad | RB F1 Team | 12.99 | 0.0% | 0.4% | 30.8% | 14.1% | 12 |
| 12 | Nico Hülkenberg | Audi | 13.24 | 0.0% | 0.3% | 28.5% | 13.1% | 11 |
| 13 | Gabriel Bortoleto | Audi | 13.53 | 0.0% | 0.2% | 24.6% | 12.8% | 13 |
| 14 | Franco Colapinto | Alpine F1 Team | 14.66 | 0.0% | 0.2% | 16.6% | 13.7% | 21 |
| 15 | Oliver Bearman | Haas F1 Team | 14.70 | 0.0% | 0.0% | 15.5% | 14.2% | 22 |
| 16 | Esteban Ocon | Haas F1 Team | 14.99 | 0.0% | 0.1% | 14.4% | 13.6% | 14 |
| 17 | Fernando Alonso | Aston Martin | 15.22 | 0.0% | 0.1% | 13.4% | 15.7% | 16 |
| 18 | Carlos Sainz | Williams | 16.12 | 0.0% | 0.1% | 10.9% | 16.5% | 18 |
| 19 | Lance Stroll | Aston Martin | 16.20 | 0.0% | 0.1% | 11.3% | 19.2% | 15 |
| 20 | Alexander Albon | Williams | 16.50 | 0.0% | 0.1% | 10.3% | 17.7% | 17 |
| 21 | Sergio Pérez | Cadillac F1 Team | 16.72 | 0.0% | 0.2% | 10.0% | 21.9% | 19 |
| 22 | Valtteri Bottas | Cadillac F1 Team | 16.81 | 0.0% | 0.1% | 10.0% | 22.9% | 20 |

The CSV and JSON contain the complete probability distribution over positions 1–22 for every driver. Actual finishing positions are included only for retrospective comparison; they were not available to the model at prediction time.
