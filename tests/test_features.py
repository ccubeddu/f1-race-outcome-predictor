from __future__ import annotations

import pandas as pd
import pytest

from f1predict.features.builder import assert_no_leakage
from f1predict.features.practice import _scheduled_pre_quali_sessions


def test_sprint_practice_after_qualifying_is_excluded() -> None:
    schedule = pd.DataFrame(
        [
            {
                "season": 2021,
                "round": 10,
                "fp1_date": "2021-07-16",
                "fp1_time": "13:30:00Z",
                "fp2_date": "2021-07-17",
                "fp2_time": "11:00:00Z",
                "fp3_date": None,
                "fp3_time": None,
                "qualifying_date": "2021-07-16",
                "qualifying_time": "17:00:00Z",
            }
        ]
    )
    eligible = _scheduled_pre_quali_sessions(schedule).set_index("session")["is_pre_quali"]
    assert bool(eligible["FP1"])
    assert not bool(eligible["FP2"])
    assert not bool(eligible["FP3"])


def test_same_day_practice_is_allowed_when_legacy_schedule_omits_times() -> None:
    schedule = pd.DataFrame(
        [
            {
                "season": 2018,
                "round": 1,
                "fp1_date": "2018-03-23",
                "fp1_time": None,
                "fp2_date": "2018-03-23",
                "fp2_time": None,
                "fp3_date": "2018-03-24",
                "fp3_time": None,
                "qualifying_date": "2018-03-24",
                "qualifying_time": None,
            }
        ]
    )
    eligible = _scheduled_pre_quali_sessions(schedule).set_index("session")["is_pre_quali"]
    assert eligible.all()


def test_leakage_guard_rejects_post_quali_feature_in_pre_mode() -> None:
    keys = pd.DataFrame({"season": [2024], "round": [1], "driver_id": ["driver"]})
    manifest = {
        "modes": {
            "pre_quali": {"features": ["driver_prior_starts", "post_grid_position"]},
            "post_quali": {"features": ["driver_prior_starts", "post_grid_position"]},
        }
    }
    with pytest.raises(ValueError, match="Pre-qualifying leakage"):
        assert_no_leakage(keys, keys, manifest)
