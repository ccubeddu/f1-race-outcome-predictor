from __future__ import annotations

import numpy as np
import pandas as pd

from f1predict.features.builder import _add_driver_history
from f1predict.features.rolling import shifted_expanding_mean, shifted_rolling_mean


def test_shifted_rolling_never_sees_current_or_other_driver() -> None:
    frame = pd.DataFrame(
        {
            "driver": ["a", "b", "a", "b", "a"],
            "finish": [10.0, 2.0, 4.0, 8.0, 1.0],
        }
    )
    result = shifted_rolling_mean(frame, "driver", "finish", 2)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 10.0
    assert result.iloc[3] == 2.0
    assert result.iloc[4] == 7.0


def test_shifted_expanding_never_sees_current() -> None:
    frame = pd.DataFrame({"driver": ["a", "a", "a"], "points": [1.0, 5.0, 100.0]})
    result = shifted_expanding_mean(frame, "driver", "points")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == 3.0


def test_driver_history_supports_an_unlabelled_future_race() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["driver", "driver", "driver"],
            "target_finish_position": [1.0, 2.0, np.nan],
            "points": [25.0, 18.0, np.nan],
            "target_dnf": [0.0, 0.0, np.nan],
            "quali_position": [2.0, 1.0, np.nan],
            "grid_position": [2.0, 1.0, np.nan],
            "completion_fraction": [1.0, 1.0, np.nan],
            "target_win": [1.0, 0.0, np.nan],
            "target_podium": [1.0, 1.0, np.nan],
        }
    )

    result = _add_driver_history(frame)

    assert result.loc[2, "driver_prior_points"] == 43.0
    assert result.loc[2, "driver_prior_wins"] == 1.0
    assert result.loc[2, "driver_finish_mean_3"] == 1.5
