from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f1predict.config import DataPaths


def audit_backtest(paths: DataPaths, mode: str) -> dict[str, Any]:
    predictions = pd.read_parquet(paths.evaluation / mode / "predictions.parquet")
    positions = sorted(
        (column for column in predictions if column.startswith("p_position_")),
        key=lambda value: int(value.rsplit("_", maxsplit=1)[-1]),
    )
    row_error = float((predictions[positions].fillna(0).sum(axis=1).sub(1).abs()).max())
    win_error = 0.0
    position_error = 0.0
    for _, race in predictions.groupby(["candidate", "season", "round"], sort=False, observed=True):
        win_error = max(win_error, abs(float(race["p_win"].sum()) - 1))
        active = positions[: len(race)]
        totals = race[active].fillna(0).sum(axis=0).to_numpy(dtype=float)
        position_error = max(position_error, float(np.max(np.abs(totals - 1))))
    training_end = pd.to_datetime(predictions["training_end"])
    race_date = pd.to_datetime(predictions["race_date"])
    chronology_violations = int(training_end.ge(race_date).sum())
    duplicate_keys = int(
        predictions.duplicated(["candidate", "season", "round", "driver_id"]).sum()
    )
    passed = (
        row_error < 1e-9
        and win_error < 1e-9
        and position_error < 1e-9
        and chronology_violations == 0
        and duplicate_keys == 0
    )
    return {
        "mode": mode,
        "passed": passed,
        "driver_probability_max_error": row_error,
        "grid_win_probability_max_error": win_error,
        "position_occupancy_max_error": position_error,
        "chronology_violations": chronology_violations,
        "duplicate_prediction_keys": duplicate_keys,
        "rows": len(predictions),
        "races": int(predictions[["season", "round"]].drop_duplicates().shape[0]),
    }
