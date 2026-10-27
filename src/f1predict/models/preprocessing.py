from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select the declared schema and normalize unsupported numeric values."""
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction data is missing model features: {missing}")
    return frame[features].replace([np.inf, -np.inf], np.nan)


def build_preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    selected = clean_features(frame, features)
    numeric = selected.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.2,
        verbose_feature_names_out=True,
    )
