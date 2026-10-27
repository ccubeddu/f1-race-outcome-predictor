from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ODDS_COLUMNS = ("decimal_odds", "american_odds", "fractional_odds")


def american_to_decimal(value: float) -> float:
    """Convert signed American odds to decimal odds."""
    if value == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def fractional_to_decimal(value: str) -> float:
    """Convert fractional odds such as ``9/4`` to decimal odds."""
    try:
        numerator, denominator = value.strip().split("/", maxsplit=1)
        decimal = 1.0 + float(numerator) / float(denominator)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid fractional odds: {value!r}") from exc
    return decimal


def load_odds(path: Path) -> pd.DataFrame:
    """Load a complete race-winner market and derive normalized fair probabilities."""
    frame = pd.read_csv(path)
    if "driver_id" not in frame:
        raise ValueError("Odds CSV must include a driver_id column")
    supplied = [column for column in ODDS_COLUMNS if column in frame]
    if len(supplied) != 1:
        raise ValueError(f"Odds CSV must include exactly one of: {', '.join(ODDS_COLUMNS)}")
    if frame["driver_id"].duplicated().any():
        duplicates = frame.loc[frame["driver_id"].duplicated(), "driver_id"].tolist()
        raise ValueError(f"Odds CSV contains duplicate drivers: {duplicates}")

    column = supplied[0]
    if column == "decimal_odds":
        decimal = pd.to_numeric(frame[column], errors="raise").astype(float)
    elif column == "american_odds":
        decimal = pd.to_numeric(frame[column], errors="raise").map(american_to_decimal)
    else:
        decimal = frame[column].map(fractional_to_decimal)
    if (~np.isfinite(decimal)).any() or (decimal <= 1.0).any():
        raise ValueError("All decimal odds must be finite and greater than 1")

    result = frame.copy()
    result["decimal_odds"] = decimal
    result["raw_implied_probability"] = 1.0 / decimal
    overround = float(result["raw_implied_probability"].sum())
    if overround <= 0:
        raise ValueError("Odds market has no positive implied probability")
    result["market_probability"] = result["raw_implied_probability"] / overround
    result.attrs["overround"] = overround
    result.attrs["margin"] = overround - 1.0
    return result


def compare_winner_market(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join model win probabilities to one complete, de-vigged winner market."""
    required = {"driver_id", "driver_name", "p_win"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {', '.join(missing)}")
    prediction_ids = set(predictions["driver_id"])
    odds_ids = set(odds["driver_id"])
    if prediction_ids != odds_ids:
        missing_odds = sorted(prediction_ids - odds_ids)
        missing_predictions = sorted(odds_ids - prediction_ids)
        details = []
        if missing_odds:
            details.append(f"missing odds for {missing_odds}")
        if missing_predictions:
            details.append(f"missing predictions for {missing_predictions}")
        raise ValueError("Driver fields do not match: " + "; ".join(details))

    columns = [
        "driver_id",
        "decimal_odds",
        "raw_implied_probability",
        "market_probability",
    ]
    compared = predictions[["driver_id", "driver_name", "p_win"]].merge(
        odds[columns], on="driver_id", validate="one_to_one"
    )
    compared["probability_delta"] = compared["p_win"] - compared["market_probability"]
    compared["model_fair_odds"] = np.where(compared["p_win"] > 0, 1.0 / compared["p_win"], np.inf)
    compared["model_expected_return"] = compared["p_win"] * compared["decimal_odds"] - 1.0
    return compared.sort_values("p_win", ascending=False, ignore_index=True)


def score_winner_market(comparison: pd.DataFrame, winner: str) -> dict[str, float | str]:
    """Score model and market winner distributions after a race is complete."""
    normalized = winner.strip().casefold()
    match = comparison[
        comparison["driver_id"].str.casefold().eq(normalized)
        | comparison["driver_name"].str.casefold().eq(normalized)
    ]
    if len(match) != 1:
        raise ValueError(f"Winner {winner!r} did not uniquely match a driver")
    actual_id = str(match.iloc[0]["driver_id"])
    target = comparison["driver_id"].eq(actual_id).astype(float).to_numpy()
    model = comparison["p_win"].to_numpy(dtype=float)
    market = comparison["market_probability"].to_numpy(dtype=float)
    actual_index = int(np.flatnonzero(target)[0])
    epsilon = 1e-15
    return {
        "winner": str(match.iloc[0]["driver_name"]),
        "model_log_loss": -math.log(max(float(model[actual_index]), epsilon)),
        "market_log_loss": -math.log(max(float(market[actual_index]), epsilon)),
        "model_brier": float(np.square(model - target).sum()),
        "market_brier": float(np.square(market - target).sum()),
    }
