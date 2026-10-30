from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write
from f1predict.models.bundle import ModelBundle
from f1predict.models.candidates import (
    RANDOM_SEED,
    fit_finish_candidate,
    fit_finish_candidates,
)
from f1predict.models.ensemble import ENSEMBLE_WEIGHTS, POLE_WIN_FACTOR
from f1predict.models.preprocessing import clean_features


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["race_date", "season", "round", "driver_id"], ignore_index=True)


def _temporal_split(
    frame: pd.DataFrame, calibration_races: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    races = frame[["season", "round", "race_date"]].drop_duplicates().sort_values("race_date")
    if len(races) < calibration_races + 20:
        raise ValueError("not enough races for the requested chronological calibration window")
    cutoff = races.iloc[-calibration_races][["season", "round"]]
    calibration = (frame["season"] > cutoff["season"]) | (
        frame["season"].eq(cutoff["season"]) & frame["round"].ge(cutoff["round"])
    )
    return frame.loc[~calibration].copy(), frame.loc[calibration].copy()


def build_dnf_classifier(*, profile: str = "full", n_jobs: int = -1) -> LGBMClassifier:
    if profile not in {"full", "backtest"}:
        raise ValueError("profile must be full or backtest")
    return LGBMClassifier(
        objective="binary",
        n_estimators=400 if profile == "full" else 120,
        learning_rate=0.03,
        num_leaves=20,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
    )


def _fit_calibrator(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
) -> tuple[LogisticRegression | None, dict[str, float]]:
    candidate = fit_finish_candidate(training, features, "lightgbm_ranker")
    train_matrix = candidate.preprocessor.transform(clean_features(training, features))
    calibration_matrix = candidate.preprocessor.transform(clean_features(calibration, features))
    model = build_dnf_classifier()
    model.fit(train_matrix, training["target_dnf"].to_numpy(dtype=int))
    raw = model.predict_proba(calibration_matrix)[:, 1]
    actual = calibration["target_dnf"].to_numpy(dtype=int)
    calibrator: LogisticRegression | None = None
    if np.unique(actual).size > 1 and np.unique(raw).size > 1:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(random_state=RANDOM_SEED)
        calibrator.fit(logit, actual)
        calibrated = calibrator.predict_proba(logit)[:, 1]
    else:
        calibrated = raw
    diagnostics = {
        "raw_mean": float(np.mean(raw)),
        "calibrated_mean": float(np.mean(calibrated)),
        "observed_rate": float(np.mean(actual)),
    }
    return calibrator, diagnostics


def _temperature_log_loss(frame: pd.DataFrame, strength: np.ndarray, temperature: float) -> float:
    work = frame[["season", "round", "target_win"]].copy()
    work["strength"] = strength
    losses: list[float] = []
    for _, race in work.groupby(["season", "round"], sort=False, observed=True):
        values = race["strength"].to_numpy(dtype=float)
        std = float(values.std())
        normalized = (values - values.mean()) / std if std > 1e-9 else np.zeros_like(values)
        probability = softmax(normalized / temperature)
        winner = race["target_win"].to_numpy(dtype=bool)
        losses.append(-float(np.log(np.clip(probability[winner][0], 1e-12, 1.0))))
    return float(np.mean(losses))


def _tune_temperature(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    kind: str = "lightgbm_ranker",
) -> tuple[float, dict[str, float]]:
    candidate = fit_finish_candidate(training, features, kind)
    strength = candidate.predict_strength(calibration)
    temperatures = np.arange(0.25, 2.01, 0.05)
    losses = {
        float(value): _temperature_log_loss(calibration, strength, float(value))
        for value in temperatures
    }
    best = min(losses, key=losses.get)
    return best, {"winner_log_loss": losses[best]}


def _atomic_joblib_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".joblib") as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_model(
    paths: DataPaths,
    mode: str,
    *,
    calibration_races: int = 30,
) -> ModelBundle:
    if mode not in {"pre_quali", "post_quali"}:
        raise ValueError("mode must be pre_quali or post_quali")
    manifest = json.loads((paths.features / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["modes"][mode]["features"]
    data = _ordered(pd.read_parquet(paths.features / f"{mode}.parquet"))
    training, calibration = _temporal_split(data, calibration_races)

    calibrator, dnf_diagnostics = _fit_calibrator(training, calibration, features)
    production_candidates = tuple(ENSEMBLE_WEIGHTS[mode])
    temperatures: dict[str, float] = {}
    temperature_diagnostics: dict[str, dict[str, float]] = {}
    for kind in production_candidates:
        temperature, diagnostics = _tune_temperature(training, calibration, features, kind)
        temperatures[kind] = temperature
        temperature_diagnostics[kind] = diagnostics
    final_candidates = fit_finish_candidates(data, features, production_candidates)
    final_ranker = final_candidates["lightgbm_ranker"]
    matrix = final_ranker.preprocessor.transform(clean_features(data, features))
    dnf_model = build_dnf_classifier()
    dnf_model.fit(matrix, data["target_dnf"].to_numpy(dtype=int))

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "practice_recipe": manifest.get("practice_recipe", "legacy_v1"),
        "training_rows": len(data),
        "training_races": int(data[["season", "round"]].drop_duplicates().shape[0]),
        "training_start": str(pd.to_datetime(data["race_date"]).min().date()),
        "training_end": str(pd.to_datetime(data["race_date"]).max().date()),
        "calibration_races": calibration_races,
        "calibration_start": str(pd.to_datetime(calibration["race_date"]).min().date()),
        "features": len(features),
        "ranker": "anchored LambdaRank/Random Forest/LightGBM regression ensemble",
        "ensemble_weights": ENSEMBLE_WEIGHTS[mode],
        "pole_win_factor": POLE_WIN_FACTOR if mode == "post_quali" else None,
        "reliability_model": "LightGBM binary classifier + chronological Platt calibration",
        "random_seed": RANDOM_SEED,
        "race_temperature": temperatures["lightgbm_ranker"],
        "race_temperatures": temperatures,
        "dnf_calibration": dnf_diagnostics,
        "temperature_calibration": temperature_diagnostics,
        "versions": {
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    bundle = ModelBundle(
        mode=mode,
        features=features,
        preprocessor=final_ranker.preprocessor,
        ranker=final_ranker.model,
        dnf_model=dnf_model,
        dnf_calibrator=calibrator,
        race_temperature=temperatures["lightgbm_ranker"],
        metadata=metadata,
        finish_models={name: candidate.model for name, candidate in final_candidates.items()},
        race_temperatures=temperatures,
        ensemble_weights=ENSEMBLE_WEIGHTS[mode],
    )
    _atomic_joblib_write(paths.models / f"{mode}.joblib", bundle)
    atomic_json_write(paths.models / f"{mode}.json", metadata)
    return bundle


def load_model(paths: DataPaths, mode: str) -> ModelBundle:
    path = paths.models / f"{mode}.joblib"
    if not path.exists():
        raise RuntimeError(f"Model does not exist: run `f1predict model train --mode {mode}`.")
    value = joblib.load(path)
    if not isinstance(value, ModelBundle):
        raise TypeError(f"Unsupported model artifact at {path}")
    return value
