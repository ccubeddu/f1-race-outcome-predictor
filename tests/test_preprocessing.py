from __future__ import annotations

import numpy as np
import pandas as pd

from f1predict.models.preprocessing import build_preprocessor


def test_preprocessor_preserves_feature_that_is_empty_in_early_history() -> None:
    training = pd.DataFrame(
        {
            "historical_form": [1.0, 2.0, 3.0],
            "future_data_source": [np.nan, np.nan, np.nan],
        }
    )
    preprocessor = build_preprocessor(training, list(training.columns))
    train_matrix = preprocessor.fit_transform(training)
    future_matrix = preprocessor.transform(
        pd.DataFrame({"historical_form": [4.0], "future_data_source": [0.25]})
    )
    assert train_matrix.shape[1] == 2
    assert future_matrix.shape[1] == 2
