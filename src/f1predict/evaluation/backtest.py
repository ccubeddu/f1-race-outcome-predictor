from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write
from f1predict.evaluation.metrics import calibration_table, evaluate_predictions
from f1predict.models.candidates import (
    RANDOM_SEED,
    FinishCandidate,
    fit_finish_candidates,
)
from f1predict.models.ensemble import (
    ENSEMBLE_WEIGHTS,
    apply_pole_win_boost,
    blend_simulations,
)
from f1predict.models.preprocessing import clean_features
from f1predict.models.training import build_dnf_classifier
from f1predict.simulation import simulate_race

CANDIDATES = ("random_forest", "lightgbm_regression", "lightgbm_ranker")


def _race_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[["season", "round", "race_date", "race_name"]]
        .drop_duplicates()
        .sort_values("race_date")
    )


def _temperature_loss(history: pd.DataFrame, temperature: float) -> float:
    losses: list[float] = []
    for _, race in history.groupby(["season", "round"], sort=False, observed=True):
        probability = softmax(race["normalized_strength"].to_numpy(dtype=float) / temperature)
        winner = race["target_win"].to_numpy(dtype=bool)
        losses.append(-float(np.log(np.clip(probability[winner][0], 1e-12, 1))))
    return float(np.mean(losses))


def _tune_temperature(history: pd.DataFrame) -> float:
    if history[["season", "round"]].drop_duplicates().shape[0] < 5:
        return 0.75
    grid = np.arange(0.2, 2.01, 0.05)
    losses = {float(value): _temperature_loss(history, float(value)) for value in grid}
    return min(losses, key=losses.get)


def _normalized_history(frame: pd.DataFrame, strength: np.ndarray) -> pd.DataFrame:
    history = frame[["season", "round", "target_win"]].copy()
    history["normalized_strength"] = strength
    history["normalized_strength"] = history.groupby(["season", "round"], observed=True)[
        "normalized_strength"
    ].transform(
        lambda values: (
            (values - values.mean()) / values.std(ddof=0) if values.std(ddof=0) > 1e-9 else 0.0
        )
    )
    return history


def _fit_dnf(
    training: pd.DataFrame,
    test: pd.DataFrame,
    candidate: FinishCandidate,
    raw_history: list[float],
    actual_history: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = candidate.preprocessor.transform(clean_features(training, candidate.features))
    test_matrix = candidate.preprocessor.transform(clean_features(test, candidate.features))
    model = build_dnf_classifier(profile="backtest")
    model.fit(train_matrix, training["target_dnf"].to_numpy(dtype=int))
    raw = model.predict_proba(test_matrix)[:, 1]
    if len(set(actual_history)) > 1 and len(set(np.round(raw_history, 8))) > 1:
        history = np.clip(np.asarray(raw_history), 1e-6, 1 - 1e-6)
        history_logit = np.log(history / (1 - history)).reshape(-1, 1)
        calibrator = LogisticRegression(random_state=RANDOM_SEED)
        calibrator.fit(history_logit, np.asarray(actual_history))
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        probability = calibrator.predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]
    else:
        probability = raw
    return raw, np.clip(probability, 0.001, 0.999)


def _bootstrap_calibration(
    history: pd.DataFrame,
    features: list[str],
    calibration_races: int = 20,
) -> tuple[dict[str, pd.DataFrame], list[float], list[int]]:
    races = _race_keys(history)
    calibration_keys = races.tail(calibration_races)[["season", "round"]]
    calibration = history.merge(calibration_keys, on=["season", "round"], how="inner")
    base = history.merge(
        calibration_keys.assign(_calibration=1), on=["season", "round"], how="left"
    )
    base = base.loc[base.pop("_calibration").isna()].copy()
    candidate_history: dict[str, pd.DataFrame] = {}
    candidates = fit_finish_candidates(base, features, CANDIDATES, profile="backtest")
    for kind, candidate in candidates.items():
        candidate_history[kind] = _normalized_history(
            calibration, candidate.predict_strength(calibration)
        )
    raw, _ = _fit_dnf(base, calibration, candidates["lightgbm_ranker"], [], [])
    return candidate_history, raw.tolist(), calibration["target_dnf"].astype(int).tolist()


def run_walk_forward(
    paths: DataPaths,
    mode: str,
    *,
    start_season: int = 2023,
    simulations: int = 5_000,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if mode not in {"pre_quali", "post_quali"}:
        raise ValueError("mode must be pre_quali or post_quali")
    manifest = json.loads((paths.features / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["modes"][mode]["features"]
    data = pd.read_parquet(paths.features / f"{mode}.parquet").sort_values(
        ["race_date", "season", "round", "driver_id"], ignore_index=True
    )
    evaluation_races = _race_keys(data.loc[data["season"].ge(start_season)])
    if evaluation_races.empty:
        raise ValueError(f"no completed races found from {start_season} onward")
    first_date = evaluation_races["race_date"].min()
    historical = data.loc[data["race_date"].lt(first_date)].copy()
    if _race_keys(historical).shape[0] < 40:
        raise ValueError("walk-forward evaluation requires at least 40 prior races")

    strength_history, raw_dnf_history, actual_dnf_history = _bootstrap_calibration(
        historical, features
    )
    outputs: list[pd.DataFrame] = []
    total = len(evaluation_races)
    for race_index, race_key in evaluation_races.reset_index(drop=True).iterrows():
        training = data.loc[data["race_date"].lt(race_key["race_date"])].copy()
        test = data.loc[
            data["season"].eq(race_key["season"]) & data["round"].eq(race_key["round"])
        ].copy()
        candidates = fit_finish_candidates(training, features, CANDIDATES, profile="backtest")
        raw_dnf, dnf_probability = _fit_dnf(
            training,
            test,
            candidates["lightgbm_ranker"],
            raw_dnf_history,
            actual_dnf_history,
        )
        raw_dnf_history.extend(raw_dnf.tolist())
        actual_dnf_history.extend(test["target_dnf"].astype(int).tolist())

        race_outputs: dict[str, pd.DataFrame] = {}
        for candidate_number, (kind, candidate) in enumerate(candidates.items()):
            strength = candidate.predict_strength(test)
            temperature = _tune_temperature(strength_history[kind])
            identifier_columns = [
                "season",
                "round",
                "race_date",
                "race_name",
                "driver_id",
                "driver_name",
                "constructor_name",
                "target_finish_position",
                "target_win",
                "target_podium",
                "target_points",
                "target_dnf",
            ]
            if mode == "post_quali":
                identifier_columns.append("post_quali_position")
            identifiers = test[identifier_columns].copy()
            result = simulate_race(
                identifiers,
                strength,
                dnf_probability,
                simulations=simulations,
                temperature=temperature,
                seed=RANDOM_SEED + race_index * 10 + candidate_number,
            )
            result["mode"] = mode
            result["candidate"] = kind
            result["temperature"] = temperature
            result["training_end"] = pd.to_datetime(training["race_date"]).max()
            outputs.append(result)
            race_outputs[kind] = result
            strength_history[kind] = pd.concat(
                [strength_history[kind], _normalized_history(test, strength)],
                ignore_index=True,
            )
        ensemble = blend_simulations(race_outputs, ENSEMBLE_WEIGHTS[mode])
        ensemble["mode"] = mode
        ensemble["candidate"] = "anchored_ensemble"
        ensemble["temperature"] = np.nan
        ensemble["training_end"] = pd.to_datetime(training["race_date"]).max()
        outputs.append(ensemble)
        if mode == "post_quali":
            pole_aware = apply_pole_win_boost(ensemble)
            pole_aware["mode"] = mode
            pole_aware["candidate"] = "pole_aware_ensemble"
            pole_aware["temperature"] = np.nan
            pole_aware["training_end"] = pd.to_datetime(training["race_date"]).max()
            outputs.append(pole_aware)
        if progress is not None:
            progress(race_index + 1, total, str(race_key["race_name"]))

    predictions = pd.concat(outputs, ignore_index=True)
    mode_dir = paths.evaluation / mode
    atomic_parquet_write(predictions, mode_dir / "predictions.parquet")
    metrics = {
        candidate: evaluate_predictions(group)
        for candidate, group in predictions.groupby("candidate", sort=False, observed=True)
    }
    calibrations = pd.concat(
        [
            calibration_table(group).assign(candidate=candidate, mode=mode)
            for candidate, group in predictions.groupby("candidate", sort=False, observed=True)
        ],
        ignore_index=True,
    )
    atomic_parquet_write(calibrations, mode_dir / "calibration.parquet")
    report: dict[str, Any] = {
        "mode": mode,
        "start_season": start_season,
        "end_season": int(evaluation_races["season"].max()),
        "races": total,
        "simulations_per_race": simulations,
        "candidates": metrics,
        "protocol": (
            "expanding-window race-by-race retraining; calibration uses only earlier "
            "out-of-fold predictions"
        ),
    }
    atomic_json_write(mode_dir / "metrics.json", report)
    return report


def rescore_saved(paths: DataPaths, mode: str) -> dict[str, Any]:
    """Recompute metrics from immutable out-of-fold driver predictions."""
    mode_dir = paths.evaluation / mode
    predictions = pd.read_parquet(mode_dir / "predictions.parquet")
    predictions = predictions.loc[
        ~predictions["candidate"].isin(["anchored_ensemble", "pole_aware_ensemble"])
    ].copy()
    ensembles: list[pd.DataFrame] = []
    for _, race in predictions.groupby(["season", "round"], sort=False, observed=True):
        simulations = {
            candidate: group.copy()
            for candidate, group in race.groupby("candidate", sort=False, observed=True)
        }
        ensemble = blend_simulations(simulations, ENSEMBLE_WEIGHTS[mode])
        ensemble["mode"] = mode
        ensemble["candidate"] = "anchored_ensemble"
        ensemble["temperature"] = np.nan
        ensembles.append(ensemble)
        if mode == "post_quali":
            pole_aware = apply_pole_win_boost(ensemble)
            pole_aware["mode"] = mode
            pole_aware["candidate"] = "pole_aware_ensemble"
            pole_aware["temperature"] = np.nan
            ensembles.append(pole_aware)
    predictions = pd.concat([predictions, *ensembles], ignore_index=True)
    report = json.loads((mode_dir / "metrics.json").read_text(encoding="utf-8"))
    if "training_end" not in predictions:
        race_dates = sorted(pd.to_datetime(predictions["race_date"].unique()))
        previous = {
            date: race_dates[index - 1] if index else pd.NaT
            for index, date in enumerate(race_dates)
        }
        historical = pd.read_parquet(paths.features / f"{mode}.parquet")
        historical_dates = sorted(pd.to_datetime(historical["race_date"].unique()))
        for date in race_dates:
            prior = [value for value in historical_dates if value < date]
            previous[date] = max(prior) if prior else pd.NaT
        predictions["training_end"] = pd.to_datetime(predictions["race_date"]).map(previous)
    atomic_parquet_write(predictions, mode_dir / "predictions.parquet")
    report["candidates"] = {
        candidate: evaluate_predictions(group)
        for candidate, group in predictions.groupby("candidate", sort=False, observed=True)
    }
    calibrations = pd.concat(
        [
            calibration_table(group).assign(candidate=candidate, mode=mode)
            for candidate, group in predictions.groupby("candidate", sort=False, observed=True)
        ],
        ignore_index=True,
    )
    atomic_parquet_write(calibrations, mode_dir / "calibration.parquet")
    atomic_json_write(mode_dir / "metrics.json", report)
    return report
