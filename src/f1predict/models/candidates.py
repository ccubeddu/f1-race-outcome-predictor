from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from f1predict.models.preprocessing import build_preprocessor, clean_features

RANDOM_SEED = 44


@dataclass
class FinishCandidate:
    name: str
    features: list[str]
    preprocessor: ColumnTransformer
    model: Any
    higher_is_better: bool

    def predict_strength(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(clean_features(frame, self.features))
        prediction = np.asarray(self.model.predict(matrix), dtype=float)
        return prediction if self.higher_is_better else -prediction


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["race_date", "season", "round", "driver_id"], ignore_index=True)


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby(["season", "round"], sort=False, observed=True).size().astype(int).tolist()


def _weighted_order(
    frame: pd.DataFrame, sample_weight: np.ndarray | None
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if sample_weight is None:
        return _ordered(frame), None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (len(frame),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weight must contain one finite positive weight per input row")
    if "__fit_weight" in frame:
        raise ValueError("Reserved fit-weight column already present")
    data = _ordered(frame.assign(__fit_weight=weights))
    return data, data.pop("__fit_weight").to_numpy()


def fit_finish_candidate(
    frame: pd.DataFrame,
    features: list[str],
    kind: str,
    *,
    profile: str = "full",
    sample_weight: np.ndarray | None = None,
    n_jobs: int = -1,
) -> FinishCandidate:
    """Fit one of the three Stage 3 finish-order candidates."""
    data, weights = _weighted_order(frame, sample_weight)
    preprocessor = build_preprocessor(data, features)
    matrix = preprocessor.fit_transform(clean_features(data, features))
    return _fit_preprocessed(data, features, kind, preprocessor, matrix, profile, weights, n_jobs)


def _fit_preprocessed(
    data: pd.DataFrame,
    features: list[str],
    kind: str,
    preprocessor: ColumnTransformer,
    matrix: Any,
    profile: str,
    sample_weight: np.ndarray | None = None,
    n_jobs: int = -1,
) -> FinishCandidate:
    if profile not in {"full", "backtest"}:
        raise ValueError("profile must be full or backtest")
    tree_counts = {
        "full": {"random_forest": 350, "lightgbm_regression": 450, "lightgbm_ranker": 500},
        "backtest": {
            "random_forest": 70,
            "lightgbm_regression": 120,
            "lightgbm_ranker": 140,
        },
    }[profile]
    finish = data["target_finish_position"].to_numpy(dtype=float)

    if kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=tree_counts[kind],
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=n_jobs,
            random_state=RANDOM_SEED,
        )
        model.fit(matrix, finish, sample_weight=sample_weight)
        higher_is_better = False
    elif kind == "lightgbm_regression":
        model = LGBMRegressor(
            objective="huber",
            n_estimators=tree_counts[kind],
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            verbosity=-1,
            n_jobs=n_jobs,
            random_state=RANDOM_SEED,
        )
        model.fit(matrix, finish, sample_weight=sample_weight)
        higher_is_better = False
    elif kind == "lightgbm_ranker":
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=tree_counts[kind],
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            verbosity=-1,
            n_jobs=n_jobs,
            random_state=RANDOM_SEED,
        )
        relevance = data["field_size"].to_numpy(dtype=int) - finish.astype(int)
        model.fit(matrix, relevance, group=_group_sizes(data), sample_weight=sample_weight)
        higher_is_better = True
    else:
        raise ValueError(f"Unknown model candidate: {kind}")
    return FinishCandidate(kind, features, preprocessor, model, higher_is_better)


def fit_finish_candidates(
    frame: pd.DataFrame,
    features: list[str],
    kinds: tuple[str, ...],
    *,
    profile: str = "full",
    sample_weight: np.ndarray | None = None,
    n_jobs: int = -1,
) -> dict[str, FinishCandidate]:
    """Fit several candidates against one identical encoded training matrix."""
    data, weights = _weighted_order(frame, sample_weight)
    preprocessor = build_preprocessor(data, features)
    matrix = preprocessor.fit_transform(clean_features(data, features))
    return {
        kind: _fit_preprocessed(
            data, features, kind, preprocessor, matrix, profile, weights, n_jobs
        )
        for kind in kinds
    }
