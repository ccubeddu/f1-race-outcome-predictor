from __future__ import annotations

import pandas as pd
import pytest

from f1predict.evaluation.odds_alignment import audit_odds_alignment


def aligned_fixture() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    row = {
        "season": 2026,
        "round": 18,
        "mode": "post_quali",
        "model_cutoff_at": "2026-10-25T19:00:00Z",
        "features_available_at": "2026-10-25T18:00:00Z",
        "odds_snapshot_at": "2026-10-25T18:59:00Z",
        "qualifying_start_at": "2026-10-24T21:00:00Z",
        "qualifying_end_at": "2026-10-24T22:10:00Z",
        "race_start_at": "2026-10-25T20:00:00Z",
        "source_url": "https://example.com/market",
        "source_market_id": "test-market",
        "timing_evidence": "Fixture only; exact archived timestamps",
    }
    predictions = pd.DataFrame(
        {
            "season": [2026],
            "round": [18],
            "mode": ["post_quali"],
            "training_end": ["2026-10-11"],
        }
    )
    odds = pd.DataFrame({"quote_at": ["2026-10-25T18:58:00Z"]})
    return row, predictions, odds


def test_aligns_fresh_quotes_to_a_common_post_quali_cutoff() -> None:
    assert audit_odds_alignment(*aligned_fixture())[0]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("odds_snapshot_at", "2026-10-23T18:59:00Z", "Features must"),
        ("features_available_at", "2026-10-25T19:01:00Z", "Features must"),
        ("model_cutoff_at", "2026-10-25T20:01:00Z", "more than"),
        ("odds_snapshot_at", "2026-10-25T18:00:00Z", "more than"),
        ("model_cutoff_at", "2026-10-25", "timezone"),
        ("mode", "pre_quali", "disagrees"),
    ],
)
def test_rejects_incompatible_or_ambiguous_timing(field: str, value: str, reason: str) -> None:
    row, predictions, odds = aligned_fixture()
    row[field] = value
    eligible, note = audit_odds_alignment(row, predictions, odds)
    assert not eligible
    assert reason in note


def test_missing_provenance_and_future_runner_quotes_are_ineligible() -> None:
    row, predictions, odds = aligned_fixture()
    assert not audit_odds_alignment({}, predictions, odds)[0]
    odds["quote_at"] = "2026-10-25T19:00:01Z"
    assert not audit_odds_alignment(row, predictions, odds)[0]


def test_post_quali_requires_odds_after_quali() -> None:
    row, predictions, odds = aligned_fixture()
    row.update(
        {
            "features_available_at": "2026-10-24T20:00:00Z",
            "model_cutoff_at": "2026-10-24T22:00:00Z",
            "odds_snapshot_at": "2026-10-24T21:59:00Z",
        }
    )
    assert "after qualifying" in audit_odds_alignment(row, predictions, odds)[1]


def test_supports_numeric_event_ids_from_a_mixed_legacy_manifest() -> None:
    row, predictions, odds = aligned_fixture()
    row.update({"season": 2026.0, "round": 18.0})
    assert audit_odds_alignment(row, predictions, odds)[0]


def test_rejects_missing_or_inconsistent_parsed_training_end() -> None:
    row, predictions, odds = aligned_fixture()
    predictions["training_end"] = "NaT"
    eligible, note = audit_odds_alignment(row, predictions, odds)
    assert not eligible
    assert "Training end" in note


def test_runner_quote_must_follow_features_even_if_snapshot_is_later() -> None:
    row, predictions, odds = aligned_fixture()
    row["features_available_at"] = "2026-10-25T18:58:30Z"
    assert "follow availability" in audit_odds_alignment(row, predictions, odds)[1]
