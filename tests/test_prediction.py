from __future__ import annotations

import pandas as pd
import pytest

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write
from f1predict.features.practice import PRACTICE_RECIPE
from f1predict.prediction import match_driver, validate_feature_recipe


def test_match_driver_accepts_name_code_or_id() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["max_verstappen", "norris"],
            "driver_code": ["VER", "NOR"],
            "driver_name": ["Max Verstappen", "Lando Norris"],
        }
    )
    assert match_driver(frame, "verstappen")["driver_code"] == "VER"
    assert match_driver(frame, "nor")["driver_id"] == "norris"


def test_match_driver_reports_unknown_name() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["max_verstappen"],
            "driver_code": ["VER"],
            "driver_name": ["Max Verstappen"],
        }
    )
    with pytest.raises(ValueError, match="was not found"):
        match_driver(frame, "senna")


def test_recipe_guard_preserves_legacy_replay_but_rejects_new_live_inputs(tmp_path) -> None:
    paths = DataPaths(tmp_path)
    atomic_json_write(paths.features / "manifest.json", {"schema_version": 1})
    validate_feature_recipe(paths, {}, "historical_replay")
    with pytest.raises(ValueError, match="Feature recipe mismatch"):
        validate_feature_recipe(paths, {}, "live_forecast")
    atomic_json_write(paths.features / "manifest.json", {"practice_recipe": PRACTICE_RECIPE})
    with pytest.raises(ValueError, match="Feature recipe mismatch"):
        validate_feature_recipe(paths, {}, "historical_replay")
    validate_feature_recipe(paths, {"practice_recipe": PRACTICE_RECIPE}, "live_forecast")
