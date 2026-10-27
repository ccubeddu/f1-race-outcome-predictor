from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1predict.config import DataPaths, find_project_root
from f1predict.data.io import atomic_json_write
from f1predict.features.live import (
    RaceEvent,
    load_race_features,
    qualifying_available,
    resolve_event,
)
from f1predict.features.practice import PRACTICE_RECIPE
from f1predict.models.ensemble import POLE_WIN_FACTOR, apply_pole_win_boost, blend_simulations
from f1predict.models.training import load_model
from f1predict.simulation import simulate_race


@dataclass
class PredictionResult:
    event: RaceEvent
    mode: str
    kind: str
    roster_source: str
    predictions: pd.DataFrame
    metadata: dict[str, Any]
    csv_path: Path | None = None
    json_path: Path | None = None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def validate_feature_recipe(paths: DataPaths, metadata: dict[str, Any], kind: str) -> None:
    """Don't silently feed a newly engineered matrix into an old trained model."""
    manifest = json.loads((paths.features / "manifest.json").read_text())
    actual = (
        PRACTICE_RECIPE if kind == "live_forecast" else manifest.get("practice_recipe", "legacy_v1")
    )
    expected = metadata.get("practice_recipe", "legacy_v1")
    if expected != actual:
        raise ValueError(
            f"Feature recipe mismatch: model={expected}, inputs={actual}. "
            "Build features and retrain matching models together in staging before promotion. "
            "Stored legacy historical forecasts remain available."
        )


def select_mode(paths: DataPaths, event: RaceEvent, requested: str) -> str:
    if requested not in {"auto", "pre_quali", "post_quali"}:
        raise ValueError("mode must be auto, pre_quali, or post_quali")
    if requested == "auto":
        return "post_quali" if qualifying_available(paths, event) else "pre_quali"
    if requested == "post_quali" and not qualifying_available(paths, event):
        raise ValueError(
            "Post-qualifying mode requested, but no qualifying classification is synced. "
            "Run the data sync after qualifying or use pre_quali."
        )
    return requested


def _save_prediction(result: PredictionResult) -> tuple[Path, Path]:
    output = find_project_root() / "outputs" / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{result.event.season}_r{result.event.round:02d}_"
        f"{_slug(result.event.race_name)}_{result.mode}"
    )
    csv_path = output / f"{stem}.csv"
    json_path = output / f"{stem}.json"
    result.predictions.to_csv(csv_path, index=False)
    atomic_json_write(json_path, prediction_payload(result))
    return csv_path, json_path


def prediction_payload(result: PredictionResult) -> dict[str, Any]:
    records = json.loads(result.predictions.to_json(orient="records", date_format="iso"))
    return {
        "event": {
            "season": result.event.season,
            "round": result.event.round,
            "race_name": result.event.race_name,
            "race_date": result.event.race_date,
        },
        "mode": result.mode,
        "prediction_kind": result.kind,
        "roster_source": result.roster_source,
        "metadata": result.metadata,
        "predictions": records,
    }


def predict_race(
    paths: DataPaths,
    *,
    season: int,
    race: str,
    mode: str = "auto",
    simulations: int = 20_000,
    seed: int = 44,
    save: bool = True,
) -> PredictionResult:
    event = resolve_event(paths, season, race)
    selected_mode = select_mode(paths, event, mode)
    features, roster_source, kind = load_race_features(paths, event, selected_mode)
    bundle = load_model(paths, selected_mode)
    validate_feature_recipe(paths, bundle.metadata, kind)
    dnf = bundle.predict_dnf_probability(features)
    identifier_columns = [
        "driver_id",
        "driver_code",
        "driver_name",
        "constructor_name",
        "driver_number",
    ]
    if selected_mode == "post_quali":
        identifier_columns.append("post_quali_position")
    identifiers = features[identifier_columns].copy()
    strengths = bundle.predict_strengths(features)
    temperatures = bundle.race_temperatures or {"lightgbm_ranker": bundle.race_temperature}
    weights = bundle.ensemble_weights or {"lightgbm_ranker": 1.0}
    member_predictions = {
        name: simulate_race(
            identifiers,
            strengths[name],
            dnf,
            simulations=simulations,
            temperature=temperatures[name],
            seed=seed + member_number,
        )
        for member_number, name in enumerate(weights)
        if weights[name] > 0
    }
    predictions = blend_simulations(member_predictions, weights)
    if selected_mode == "post_quali":
        predictions = apply_pole_win_boost(predictions)
    result = PredictionResult(
        event=event,
        mode=selected_mode,
        kind=kind,
        roster_source=roster_source,
        predictions=predictions,
        metadata={
            "model": (
                "Pole-aware anchored ensemble + calibrated DNF + Monte Carlo"
                if selected_mode == "post_quali"
                else "Anchored ranking ensemble + calibrated DNF + Monte Carlo"
            ),
            "model_training_end": bundle.metadata["training_end"],
            "simulations": simulations,
            "seed": seed,
            "race_temperature": bundle.race_temperature,
            "race_temperatures": temperatures,
            "ensemble_weights": weights,
            "pole_win_factor": POLE_WIN_FACTOR if selected_mode == "post_quali" else None,
        },
    )
    if save:
        result.csv_path, result.json_path = _save_prediction(result)
    return result


def match_driver(predictions: pd.DataFrame, query: str) -> pd.Series:
    needle = query.strip().casefold()
    match = pd.Series(False, index=predictions.index)
    for column in ("driver_id", "driver_code", "driver_name"):
        match |= (
            predictions[column]
            .fillna("")
            .astype(str)
            .str.casefold()
            .str.contains(needle, regex=False)
        )
    candidates = predictions.loc[match]
    if len(candidates) != 1:
        names = ", ".join(candidates["driver_name"].astype(str).tolist())
        if candidates.empty:
            names = ", ".join(predictions["driver_name"].astype(str).tolist())
            raise ValueError(f"Driver {query!r} was not found. Grid: {names}")
        raise ValueError(f"Driver {query!r} is ambiguous. Matches: {names}")
    return candidates.iloc[0]
