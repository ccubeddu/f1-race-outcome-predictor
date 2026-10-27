from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from f1predict.models.preprocessing import clean_features


@dataclass
class ModelBundle:
    """Serializable ranking and reliability models sharing one feature contract."""

    mode: str
    features: list[str]
    preprocessor: ColumnTransformer
    ranker: Any
    dnf_model: Any
    dnf_calibrator: Any | None
    race_temperature: float
    metadata: dict[str, Any]
    finish_models: dict[str, Any] | None = None
    race_temperatures: dict[str, float] | None = None
    ensemble_weights: dict[str, float] | None = None

    def transform(self, frame: pd.DataFrame) -> Any:
        return self.preprocessor.transform(clean_features(frame, self.features))

    def predict_strength(self, frame: pd.DataFrame) -> np.ndarray:
        """Higher means stronger expected race performance."""
        return np.asarray(self.ranker.predict(self.transform(frame)), dtype=float)

    def predict_strengths(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return higher-is-better scores for every production ensemble member."""
        matrix = self.transform(frame)
        models = self.finish_models or {"lightgbm_ranker": self.ranker}
        strengths: dict[str, np.ndarray] = {}
        for name, model in models.items():
            values = np.asarray(model.predict(matrix), dtype=float)
            strengths[name] = values if name == "lightgbm_ranker" else -values
        return strengths

    def predict_dnf_probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.dnf_model.predict_proba(self.transform(frame))[:, 1], dtype=float)
        if self.dnf_calibrator is not None:
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            raw = self.dnf_calibrator.predict_proba(logit)[:, 1]
        return np.clip(raw, 0.001, 0.999)

    def feature_importance(self) -> pd.DataFrame:
        transformed = self.preprocessor.get_feature_names_out()
        gains = self.ranker.booster_.feature_importance(importance_type="gain")
        source_names: list[str] = []
        categorical = set(self.preprocessor.transformers_[1][2])
        for name in transformed:
            clean = name.split("__", maxsplit=1)[-1]
            source = clean
            for feature in categorical:
                if clean == feature or clean.startswith(f"{feature}_"):
                    source = feature
                    break
            source_names.append(source)
        importance = pd.DataFrame({"feature": source_names, "gain": gains})
        importance = importance.groupby("feature", as_index=False, observed=True)["gain"].sum()
        total = importance["gain"].sum()
        importance["importance"] = importance["gain"].div(total if total else 1.0)
        return importance.sort_values("gain", ascending=False, ignore_index=True)
