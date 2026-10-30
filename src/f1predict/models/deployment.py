"""One full-size fit/calibrate/predict recipe for research backtests and saved candidates.

Legacy production artifacts are deliberately not migrated by this module. A calibration
record is admitted only after its forecast is frozen; the next race uses earlier records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from f1predict.evaluation.grid_simulation import position_matrix, prediction_frame
from f1predict.models.candidates import FinishCandidate, fit_finish_candidates
from f1predict.models.ensemble import ENSEMBLE_WEIGHTS, apply_pole_win_boost, blend_simulations
from f1predict.models.preprocessing import build_preprocessor, clean_features
from f1predict.models.training import build_dnf_classifier

RECIPE = "full_oof_deployment_v1"
TEMPERATURES = (0.3, 0.45, 0.65, 0.85, 1.1, 1.45, 1.9)
MEMBERS = tuple(ENSEMBLE_WEIGHTS["pre_quali"])
VERSIONS = {
    name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")
}
IDENTIFIERS = [
    "season",
    "round",
    "race_date",
    "race_name",
    "driver_id",
    "driver_name",
    "constructor_name",
]


@dataclass(frozen=True)
class DeploymentConfig:
    start_year: int = 2018
    half_life_days: float | None = None
    ratings: bool = False
    calibration_races: int = 30
    n_jobs: int = 1

    @property
    def name(self) -> str:
        family = "ratings" if self.ratings else "base"
        weight = "uniform" if self.half_life_days is None else "decay"
        return f"{family}_{self.start_year}_{weight}"

    def to_dict(self) -> dict:
        return asdict(self)


def default_configurations() -> list[DeploymentConfig]:
    return [
        DeploymentConfig(year, half_life, ratings)
        for ratings, years in ((False, (2018, 2014, 2010)), (True, (2010,)))
        for year in years
        for half_life in (None, 730.5)
    ]


def training_rows(
    data: pd.DataFrame, cutoff: pd.Timestamp, config: DeploymentConfig
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if config.start_year < 2010 or config.calibration_races < 5:
        raise ValueError("Invalid deployment configuration")
    if config.half_life_days is not None and config.half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    rows = data.loc[
        data.season.ge(config.start_year) & pd.to_datetime(data.race_date).lt(cutoff)
    ].sort_values(["race_date", "season", "round", "driver_id"], ignore_index=True)
    if rows.empty or not pd.to_datetime(rows.race_date).max() < cutoff:
        raise ValueError("No strictly earlier training rows")
    if rows[["season", "round"]].drop_duplicates().shape[0] < 40:
        raise ValueError("Need at least 40 earlier training races")
    if config.half_life_days is None:
        return rows, None
    age = (cutoff - pd.to_datetime(rows.race_date)).dt.total_seconds().to_numpy() / 86400
    weights = np.exp2(-age / config.half_life_days)
    # Normalize for comparable regularization. All entrants of an event get equal weight.
    return rows, weights / weights.mean()


def calibration_parameters(
    history: list[dict], cutoff: pd.Timestamp, config: DeploymentConfig, mode: str
) -> tuple[Any, dict[str, float], dict]:
    dates = [pd.Timestamp(record["race_date"]) for record in history]
    if any(date >= cutoff for date in dates) or dates != sorted(set(dates)):
        raise ValueError("Calibration records must be unique, ordered and strictly earlier")
    recent = history[-config.calibration_races :]
    calibrator = None
    if recent:
        raw = np.concatenate([r["raw_dnf"] for r in recent])
        actual = np.concatenate([r["actual_dnf"] for r in recent])
        if len(raw) >= 50 and np.unique(actual).size == 2 and np.unique(raw).size > 1:
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            calibrator = LogisticRegression(random_state=44).fit(
                np.log(clipped / (1 - clipped))[:, None], actual.astype(int)
            )
    temperatures = {}
    for kind in ENSEMBLE_WEIGHTS[mode]:
        temperatures[kind] = (
            TEMPERATURES[int(np.argmin(np.mean([r["losses"][kind] for r in recent], axis=0)))]
            if len(recent) >= 5
            else 0.85
        )
    return (
        calibrator,
        temperatures,
        {
            "races": len(recent),
            "start": str(recent[0]["race_date"]) if recent else None,
            "end": str(recent[-1]["race_date"]) if recent else None,
            "policy": f"last {config.calibration_races} earlier OOF races; simulated winner loss",
        },
    )


@dataclass
class DeploymentBundle:
    mode: str
    config: DeploymentConfig
    models: dict[str, FinishCandidate]
    dnf_features: list[str]
    dnf_preprocessor: Any
    dnf_model: Any
    dnf_calibrator: Any
    temperatures: dict[str, float]
    metadata: dict

    def raw_scores(self, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if frame.duplicated("driver_id").any() or frame.race_date.nunique() != 1:
            raise ValueError("Prediction requires one event with unique drivers")
        if not pd.Timestamp(self.metadata["training_end"]) < pd.Timestamp(frame.race_date.iloc[0]):
            raise ValueError("Prediction target must be strictly after model training")
        strengths = {kind: model.predict_strength(frame) for kind, model in self.models.items()}
        matrix = self.dnf_preprocessor.transform(clean_features(frame, self.dnf_features))
        return strengths, self.dnf_model.predict_proba(matrix)[:, 1]

    def calibrate_dnf(self, raw: np.ndarray) -> np.ndarray:
        if self.dnf_calibrator is not None:
            raw = np.clip(raw, 1e-6, 1 - 1e-6)
            raw = self.dnf_calibrator.predict_proba(np.log(raw / (1 - raw))[:, None])[:, 1]
        return np.clip(raw, 0.001, 0.999)

    def predict(
        self, frame: pd.DataFrame, *, simulations: int = 4096, seed: int = 44
    ) -> pd.DataFrame:
        strengths, raw = self.raw_scores(frame)
        return forecast_scores(
            frame,
            strengths,
            self.calibrate_dnf(raw),
            self.temperatures,
            self.mode,
            simulations=simulations,
            seed=seed,
        )


def fit_deployment_bundle(
    data: pd.DataFrame,
    features: list[str],
    dnf_features: list[str],
    mode: str,
    config: DeploymentConfig,
    cutoff: pd.Timestamp,
    history: list[dict],
) -> DeploymentBundle:
    if mode not in ENSEMBLE_WEIGHTS:
        raise ValueError("Unknown prediction cutoff")
    if any(c.startswith(("target_", "label_")) for c in [*features, *dnf_features]):
        raise ValueError("Outcome labels are not model inputs")
    if mode == "pre_quali" and any(c.startswith("post_") for c in [*features, *dnf_features]):
        raise ValueError("Pre-quali models cannot use post-quali features")
    training, weights = training_rows(data, cutoff, config)
    calibrator, temperatures, calibration = calibration_parameters(history, cutoff, config, mode)
    models = fit_finish_candidates(
        training,
        features,
        tuple(ENSEMBLE_WEIGHTS[mode]),
        profile="full",
        sample_weight=weights,
        n_jobs=config.n_jobs,
    )
    if features == dnf_features:
        preprocessor = models["lightgbm_ranker"].preprocessor
    else:
        preprocessor = build_preprocessor(training, dnf_features)
        preprocessor.fit(clean_features(training, dnf_features))
    dnf_model = build_dnf_classifier(profile="full", n_jobs=config.n_jobs)
    dnf_model.fit(
        preprocessor.transform(clean_features(training, dnf_features)),
        training.target_dnf.to_numpy(dtype=int),
        sample_weight=weights,
    )
    metadata = {
        "schema_version": 1,
        "versions": VERSIONS,
        "recipe": RECIPE,
        "mode": mode,
        "configuration": config.to_dict(),
        "profile": "full",
        "tree_counts": {"ranker": 500, "forest": 350, "regression": 450, "dnf": 400},
        "training_start": str(training.race_date.min()),
        "training_end": str(training.race_date.max()),
        "target_date": str(cutoff),
        "training_rows": len(training),
        "training_races": len(training[["season", "round"]].drop_duplicates()),
        "features": features,
        "dnf_features": dnf_features,
        "calibration": calibration,
        "temperatures": temperatures,
        "weight_min": float(weights.min()) if weights is not None else 1.0,
        "weight_max": float(weights.max()) if weights is not None else 1.0,
        "preprocessing_weighted": False,
        "pole_win_factor": 1.25 if mode == "post_quali" else None,
    }
    return DeploymentBundle(
        mode,
        config,
        models,
        dnf_features,
        preprocessor,
        dnf_model,
        calibrator,
        temperatures,
        metadata,
    )


def forecast_scores(
    frame: pd.DataFrame,
    strengths: dict[str, np.ndarray],
    dnf: np.ndarray,
    temperatures: dict[str, float],
    mode: str,
    *,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Label-free prediction shared by serialized bundles and cached-score sensitivity tests."""
    identifiers = frame[[*IDENTIFIERS, *(["post_quali_position"] if mode == "post_quali" else [])]]
    members = {
        kind: prediction_frame(
            identifiers,
            position_matrix(
                strengths[kind],
                dnf,
                temperature=temperatures[kind],
                simulations=simulations,
                seed=seed + i,
            ),
            dnf,
        )
        for i, kind in enumerate(ENSEMBLE_WEIGHTS[mode])
    }
    prediction = blend_simulations(members, ENSEMBLE_WEIGHTS[mode])
    if mode == "post_quali":
        prediction = apply_pole_win_boost(prediction)
    return prediction


def calibration_record(
    frame: pd.DataFrame,
    strengths: dict[str, np.ndarray],
    raw_dnf: np.ndarray,
    dnf: np.ndarray,
    *,
    simulations: int,
    seed: int,
) -> dict:
    """Call only AFTER freezing a forecast. Current labels affect only later races."""
    winner = frame.target_win.to_numpy(dtype=bool)
    if winner.sum() != 1:
        raise ValueError("Completed calibration race must have exactly one winner")
    losses = {
        kind: [
            -float(
                np.log(
                    position_matrix(
                        strengths[kind], dnf, temperature=t, simulations=simulations, seed=seed + i
                    )[winner, 0][0]
                )
            )
            for t in TEMPERATURES
        ]
        for i, kind in enumerate(MEMBERS)
    }
    return {
        "race_date": str(frame.race_date.iloc[0]),
        "raw_dnf": raw_dnf.tolist(),
        "actual_dnf": frame.target_dnf.astype(int).tolist(),
        "losses": losses,
    }
