from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUTCOMES = {
    "win": ("p_win", "target_win"),
    "podium": ("p_podium", "target_podium"),
    "points": ("p_points", "target_points"),
    "dnf": ("p_dnf", "target_dnf"),
}


def _binary_metrics(probability: pd.Series, actual: pd.Series) -> dict[str, float]:
    predicted = np.clip(probability.to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    observed = actual.to_numpy(dtype=float)
    base_rate = float(np.mean(observed))
    brier = float(np.mean((predicted - observed) ** 2))
    null_brier = base_rate * (1 - base_rate)
    return {
        "base_rate": base_rate,
        "log_loss": float(
            np.mean(-(observed * np.log(predicted) + (1 - observed) * np.log(1 - predicted)))
        ),
        "brier": brier,
        "null_brier": null_brier,
        "brier_skill": float(1 - brier / null_brier) if null_brier > 0 else 0.0,
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    race_metrics: list[dict[str, float]] = []
    position_columns = sorted(
        (column for column in predictions if column.startswith("p_position_")),
        key=lambda value: int(value.rsplit("_", maxsplit=1)[-1]),
    )
    for _, race in predictions.groupby(["season", "round"], sort=False, observed=True):
        actual = race["target_finish_position"].to_numpy(dtype=float)
        predicted = race["predicted_position"].to_numpy(dtype=float)
        row: dict[str, float] = {
            "spearman": float(spearmanr(actual, predicted).statistic),
            "top3_accuracy": float(
                len(
                    set(race.nsmallest(3, "predicted_position").driver_id)
                    & set(race.loc[race.target_podium.eq(1), "driver_id"])
                )
                / 3
            ),
            "top10_accuracy": float(
                len(
                    set(race.nsmallest(10, "predicted_position").driver_id)
                    & set(race.loc[race.target_finish_position.le(10), "driver_id"])
                )
                / min(10, len(race))
            ),
            "winner_accuracy": float(race.nsmallest(1, "predicted_position")["target_win"].iloc[0]),
            "position_mae": float(np.mean(np.abs(actual - predicted))),
        }
        probabilities = race[position_columns].fillna(0).to_numpy(dtype=float)
        actual_index = actual.astype(int) - 1
        chosen = probabilities[np.arange(len(race)), actual_index]
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(race)), actual_index] = 1
        row["position_log_loss"] = float(np.mean(-np.log(np.clip(chosen, 1e-9, 1))))
        row["position_brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        race_metrics.append(row)

    per_race = pd.DataFrame(race_metrics)
    result: dict[str, Any] = {
        "races": len(per_race),
        "driver_predictions": len(predictions),
        **{column: float(per_race[column].mean()) for column in per_race},
        "probabilities": {},
    }
    for label, (probability, target) in OUTCOMES.items():
        result["probabilities"][label] = _binary_metrics(
            predictions[probability], predictions[target]
        )
    return result


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    edges = np.linspace(0, 1, bins + 1)
    for outcome, (probability, target) in OUTCOMES.items():
        work = predictions[[probability, target]].copy()
        work["bin"] = pd.cut(
            work[probability], edges, include_lowest=True, labels=False, duplicates="drop"
        )
        for bin_number, group in work.dropna(subset=["bin"]).groupby("bin", observed=True):
            rows.append(
                {
                    "outcome": outcome,
                    "bin": int(bin_number),
                    "predicted": float(group[probability].mean()),
                    "observed": float(group[target].mean()),
                    "count": len(group),
                }
            )
    return pd.DataFrame(rows)
