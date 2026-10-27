"""Eligibility checks for forecasts compared at the same information cutoff.

Passing these checks verifies the supplied timing metadata, not that a model is an
untouched holdout or that a historic last-traded price could have been executed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

TIME_FIELDS = (
    "model_cutoff_at",
    "features_available_at",
    "odds_snapshot_at",
    "qualifying_start_at",
    "qualifying_end_at",
    "race_start_at",
)


def utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    """Require an explicit timezone; local or date-only timestamps are ambiguous."""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an ISO timestamp with a timezone") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must include a timezone, e.g. 2026-10-25T19:00:00Z")
    return stamp.tz_convert("UTC")


def audit_odds_alignment(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    max_gap_seconds: int = 300,
) -> tuple[bool, str]:
    """Fail closed when common timing, event identity, or provenance is unknown."""
    missing = [
        name
        for name in (
            *TIME_FIELDS,
            "season",
            "round",
            "mode",
            "source_url",
            "source_market_id",
            "timing_evidence",
        )
        if name not in row or pd.isna(row[name]) or str(row[name]).strip() == ""
    ]
    if missing:
        return False, "Missing timing/provenance fields: " + ", ".join(missing)
    try:
        stamps = {name: utc_timestamp(row[name], name) for name in TIME_FIELDS}
    except ValueError as exc:
        return False, str(exc)
    mode = str(row["mode"])
    if mode not in {"pre_quali", "post_quali"}:
        return False, "mode must be pre_quali or post_quali"
    for name in ("season", "round", "mode", "training_end"):
        if name not in predictions or predictions[name].isna().any():
            return False, f"Prediction artifact lacks {name} provenance"
    for name in ("season", "round", "mode"):
        values = predictions[name].unique()
        if name in {"season", "round"}:
            try:
                expected = float(row[name])
                actual = [float(value) for value in values]
            except (TypeError, ValueError):
                return False, f"Invalid manifest or artifact {name}"
            consistent = len(actual) == 1 and expected.is_integer() and actual[0] == expected
        else:
            consistent = len(values) == 1 and str(values[0]) == str(row[name])
        if not consistent:
            return False, f"Prediction artifact disagrees with manifest {name}"
    if not (stamps["qualifying_start_at"] < stamps["qualifying_end_at"] < stamps["race_start_at"]):
        return False, "Session times must satisfy qualifying start < end < race start"
    model_at = stamps["model_cutoff_at"]
    book_at = stamps["odds_snapshot_at"]
    if not (stamps["features_available_at"] <= book_at <= model_at):
        return False, "Features must be available by the odds snapshot and common model cutoff"
    if (model_at - book_at).total_seconds() > max_gap_seconds:
        return False, f"Odds snapshot is more than {max_gap_seconds}s before model cutoff"
    if model_at >= stamps["race_start_at"]:
        return False, "Forecast cutoff must be before the race starts"
    if mode == "pre_quali" and model_at >= stamps["qualifying_start_at"]:
        return False, "pre_quali cutoff must be before qualifying starts"
    if mode == "post_quali" and book_at < stamps["qualifying_end_at"]:
        return False, "post_quali odds must be captured after qualifying ends"
    try:
        training_end = pd.to_datetime(predictions["training_end"], utc=True, errors="raise")
    except (ValueError, TypeError):
        return False, "Invalid training_end in prediction artifact"
    if training_end.isna().any() or training_end.nunique() != 1:
        return False, "Training end must be present and consistent across the prediction artifact"
    # Existing backtests record dates, not result-publication times. Only earlier
    # calendar days are unambiguous; same-day training requires richer provenance.
    if training_end.max().normalize() >= book_at.normalize():
        return False, "Training results must precede the forecast calendar day"
    if "quote_at" not in odds or odds["quote_at"].isna().any():
        return False, "Odds artifact needs a quote_at timestamp for every driver"
    try:
        quotes = odds["quote_at"].map(lambda value: utc_timestamp(value, "quote_at"))
    except ValueError as exc:
        return False, str(exc)
    if (quotes < stamps["features_available_at"]).any():
        return False, "Every runner quote must follow availability of all model features"
    if mode == "post_quali" and (quotes < stamps["qualifying_end_at"]).any():
        return False, "Every post_quali runner quote must follow qualifying completion"
    if (quotes > book_at).any() or ((book_at - quotes).dt.total_seconds() > max_gap_seconds).any():
        return False, "Runner quotes are after the snapshot or too stale for the selected cutoff"
    return True, "Timing aligned; retrospective model selection must be assessed separately"
