from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f1predict.bookmakers import (
    american_to_decimal,
    compare_winner_market,
    fractional_to_decimal,
    load_odds,
    score_winner_market,
)


def test_odds_conversions() -> None:
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert fractional_to_decimal("9/4") == 3.25


def test_compare_and_score_complete_market(tmp_path: Path) -> None:
    path = tmp_path / "odds.csv"
    pd.DataFrame(
        {"driver_id": ["a", "b"], "american_odds": [-110, -110]}
    ).to_csv(path, index=False)
    odds = load_odds(path)
    predictions = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "driver_name": ["Driver A", "Driver B"],
            "p_win": [0.7, 0.3],
        }
    )
    comparison = compare_winner_market(predictions, odds)
    score = score_winner_market(comparison, "Driver A")
    assert odds.attrs["margin"] == pytest.approx(1 / 21)
    assert comparison["market_probability"].sum() == pytest.approx(1.0)
    assert score["model_log_loss"] < score["market_log_loss"]


def test_compare_rejects_incomplete_grid(tmp_path: Path) -> None:
    path = tmp_path / "odds.csv"
    pd.DataFrame({"driver_id": ["a"], "decimal_odds": [2.0]}).to_csv(path, index=False)
    odds = load_odds(path)
    predictions = pd.DataFrame(
        {"driver_id": ["a", "b"], "driver_name": ["A", "B"], "p_win": [0.5, 0.5]}
    )
    with pytest.raises(ValueError, match="missing odds"):
        compare_winner_market(predictions, odds)
