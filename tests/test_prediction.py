from __future__ import annotations

import pandas as pd
import pytest

from f1predict.prediction import match_driver


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
