"""Offline, as-of import of Betfair historical winner-market last-traded prices.

BASIC files contain one-minute LTP updates and market definitions, not executable
back/lay offers. The importer never downloads data or infers an event/cutoff.
Format: https://github.com/betfair/stream-api-sample-code/blob/master/ESASwaggerSchema.json
Packages: https://historicdata.betfair.com/Betfair-Historical-Data-Feed-Specification.pdf
"""

from __future__ import annotations

import bz2
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

WINNER_MARKET_TYPES = frozenset({"WIN", "WINNER", "OUTRIGHT_WINNER"})


def _utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _selection_id(value: Any) -> str:
    if type(value) is not int or value <= 0:
        raise ValueError("Betfair selection id must be a positive integer")
    return str(value)


def _validate_mapping(driver_map: dict[str, str]) -> None:
    if len(driver_map) < 2:
        raise ValueError("driver_map must explicitly map the complete grid (at least two drivers)")
    for selection, driver in driver_map.items():
        if not isinstance(selection, str) or not selection.isdecimal() or int(selection) <= 0:
            raise ValueError("driver_map keys must be positive selection IDs encoded as strings")
        if str(int(selection)) != selection:
            raise ValueError("driver_map selection IDs must not have leading zeros")
        if not isinstance(driver, str) or not driver.strip() or driver != driver.strip():
            raise ValueError("driver_map values must be nonempty canonical driver IDs")
    if len(set(driver_map.values())) != len(driver_map):
        raise ValueError("driver_map must map each selection to a distinct canonical driver ID")


def _validate_definition(
    definition: dict[str, Any],
    driver_map: dict[str, str],
    cutoff: pd.Timestamp,
    expected_market_type: str,
) -> tuple[dict[str, dict[str, Any]], pd.Timestamp]:
    if (
        definition.get("marketType") != expected_market_type
        or definition.get("bettingType") != "ODDS"
    ):
        raise ValueError(
            f"Only explicitly identified {expected_market_type}/ODDS markets are supported"
        )
    if type(definition.get("numberOfWinners")) is not int or definition["numberOfWinners"] != 1:
        raise ValueError("WIN market must explicitly have numberOfWinners=1")
    if definition.get("inPlay") is not False:
        raise ValueError("Market must explicitly be pre-race (inPlay=false) at the cutoff")
    if definition.get("status") != "OPEN":
        raise ValueError("Market must be OPEN at the cutoff; suspended/closed markets are rejected")
    if definition.get("complete") is not True:
        raise ValueError("Market definition must explicitly declare a complete grid")
    market_time = _utc_timestamp(definition.get("marketTime"), "marketTime")
    if cutoff >= market_time:
        raise ValueError("Cutoff must precede the scheduled marketTime for a pre-race snapshot")
    raw_runners = definition.get("runners")
    if not isinstance(raw_runners, list) or not raw_runners:
        raise ValueError("Market definition must contain all runners")
    runners: dict[str, dict[str, Any]] = {}
    for runner in raw_runners:
        if not isinstance(runner, dict):
            raise ValueError("Every market runner must be an object")
        selection = _selection_id(runner.get("id"))
        if selection in runners:
            raise ValueError(f"Duplicate runner definition for selection {selection}")
        if runner.get("status") != "ACTIVE":
            raise ValueError(f"Runner {selection} must be ACTIVE at the cutoff")
        if runner.get("hc") not in (None, 0):
            raise ValueError("Handicap runners are unsupported in a WIN market")
        runners[selection] = runner
    if set(runners) != set(driver_map):
        missing = sorted(set(driver_map) - set(runners))
        unmapped = sorted(set(runners) - set(driver_map))
        raise ValueError(f"Incomplete grid/driver mapping: missing={missing}, unmapped={unmapped}")
    count = definition.get("numberOfActiveRunners")
    if type(count) is not int or count != len(runners):
        raise ValueError("numberOfActiveRunners must match the complete mapped grid")
    return runners, market_time


def import_betfair_snapshot(
    stream_path: Path,
    driver_map: dict[str, str],
    market_id: str,
    cutoff: pd.Timestamp | str,
    max_age_seconds: int = 3600,
    *,
    expected_market_type: str = "WIN",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct one complete pre-race winner market at an explicit UTC cutoff.

    Accept an extracted NDJSON stream or a single ``.bz2`` stream (not tar).
    Only messages published at or before the cutoff may update the cache. Each
    runner's ``quote_at`` is its last non-null LTP *publication* time, not the
    unknown time of the actual trade. Mapping keys are Betfair selection IDs;
    values are canonical driver IDs. The mapping must cover the whole market.
    ``expected_market_type`` must match the source definition exactly; use an
    explicit ``WINNER`` or ``OUTRIGHT_WINNER`` when that is the race market's
    verified code. These codes are not proof of F1 race/event identity, which the
    caller must establish separately; real F1 archive coverage is not assumed.

    Market definitions are replacement objects; runner prices are deltas.
    Full images reset both caches. Segmentation, selected-market timestamp
    regressions, stale prices, and incomplete/unsafe market states fail closed.
    No settled result is inspected and no outcome is inferred from future data.
    """
    stream_path = Path(stream_path)
    _validate_mapping(driver_map)
    if not isinstance(market_id, str) or not market_id.strip() or market_id != market_id.strip():
        raise ValueError("An explicit nonempty market_id is required")
    cutoff_at = _utc_timestamp(cutoff, "cutoff")
    if type(max_age_seconds) is not int or max_age_seconds < 0:
        raise ValueError("max_age_seconds must be a nonnegative integer")
    if expected_market_type not in WINNER_MARKET_TYPES:
        raise ValueError(
            "expected_market_type must be an explicit winner code: WIN, WINNER, OUTRIGHT_WINNER"
        )
    source_hash = hashlib.sha256()
    with stream_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            source_hash.update(chunk)

    definition: dict[str, Any] | None = None
    quotes: dict[str, tuple[float, pd.Timestamp]] = {}
    asof: pd.Timestamp | None = None
    applied_changes = 0
    open_stream = bz2.open if stream_path.suffix.lower() == ".bz2" else open
    with open_stream(stream_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid NDJSON at line {line_number}") from exc
            if not isinstance(message, dict):
                raise ValueError(f"Expected a stream object at line {line_number}")
            if message.get("op") != "mcm":
                raise ValueError(
                    f"Unsupported stream operation at line {line_number}; expected mcm"
                )
            published_ms = message.get("pt")
            if type(published_ms) is not int or published_ms < 0:
                raise ValueError(f"Missing/invalid published time pt at line {line_number}")
            try:
                published_at = pd.Timestamp(published_ms, unit="ms", tz="UTC")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid published time pt at line {line_number}") from exc
            # Do not stop at a future message: source order need not guarantee
            # monotonic pt. Future updates must never affect the as-of state.
            if published_at > cutoff_at:
                continue
            if message.get("segmentType") is not None:
                raise ValueError("Segmented stream messages are unsupported; use unsegmented data")
            if message.get("status") is not None:
                raise ValueError("Unreliable stream status before cutoff; snapshot rejected")
            changes = message.get("mc")
            if changes is None and message.get("ct") == "HEARTBEAT":
                continue
            if not isinstance(changes, list):
                raise ValueError(f"Missing/invalid market changes at line {line_number}")
            for change in changes:
                if not isinstance(change, dict) or not isinstance(change.get("id"), str):
                    raise ValueError(f"Invalid market change at line {line_number}")
                if change["id"] != market_id:
                    continue
                if asof is not None and published_at < asof:
                    raise ValueError("Selected market timestamps regress in source order")
                if change.get("img") is True:
                    definition, quotes = None, {}
                elif change.get("img") is not None and change["img"] is not False:
                    raise ValueError("Market img must be a boolean or null")
                if change.get("marketDefinition") is not None:
                    if not isinstance(change["marketDefinition"], dict):
                        raise ValueError("marketDefinition must be an object")
                    definition = change["marketDefinition"]
                runner_changes = change.get("rc")
                if runner_changes is not None:
                    if not isinstance(runner_changes, list):
                        raise ValueError("Runner changes rc must be a list")
                    for runner in runner_changes:
                        if not isinstance(runner, dict):
                            raise ValueError("Runner change must be an object")
                        selection = _selection_id(runner.get("id"))
                        if selection not in driver_map:
                            raise ValueError(f"Unmapped selection {selection} before cutoff")
                        if runner.get("hc") not in (None, 0):
                            raise ValueError("Handicap runner changes are unsupported")
                        price = runner.get("ltp")
                        if price is None:  # null/absent means unchanged, not a fresh quote.
                            continue
                        if (
                            isinstance(price, bool)
                            or not isinstance(price, (int, float))
                            or not math.isfinite(price)
                            or not 1 < price <= 1000
                        ):
                            raise ValueError(f"Invalid decimal last-traded price for {selection}")
                        quotes[selection] = (float(price), published_at)
                asof = published_at
                applied_changes += 1

    if definition is None or asof is None:
        raise ValueError(f"No marketDefinition for market {market_id} at or before cutoff")
    runners, market_time = _validate_definition(
        definition, driver_map, cutoff_at, expected_market_type
    )
    if set(quotes) != set(runners):
        missing_quotes = sorted(set(runners) - set(quotes))
        raise ValueError(
            f"Complete grid requires a last-traded price for every runner: {missing_quotes}"
        )
    rows = []
    for selection, driver in driver_map.items():
        price, quote_at = quotes[selection]
        age = (cutoff_at - quote_at).total_seconds()
        if age > max_age_seconds:
            raise ValueError(f"Stale last-traded price for {driver}: {age:g}s before cutoff")
        rows.append({"driver_id": driver, "decimal_odds": price, "quote_at": quote_at.isoformat()})
    frame = pd.DataFrame(rows).sort_values("driver_id", ignore_index=True)
    metadata = {
        "source": "betfair_historical_stream",
        "source_file": stream_path.name,
        "source_sha256": source_hash.hexdigest(),
        "source_hash_scope": "raw_file_bytes",
        "market_id": market_id,
        "market_type": definition["marketType"],
        "expected_market_type": expected_market_type,
        "market_name": definition.get("name"),
        "event_name": definition.get("eventName"),
        "market_time": market_time.isoformat(),
        "cutoff": cutoff_at.isoformat(),
        "asof": asof.isoformat(),
        "price_type": "last_traded",
        "quote_timestamp_type": "stream_publication_time",
        "runner_count": len(frame),
        "max_age_seconds": max_age_seconds,
        "applied_market_changes": applied_changes,
        "caveat": (
            "BASIC last-traded prices are not executable back/lay offers. quote_at is the "
            "publication time of the last supplied LTP, not the actual trade timestamp; "
            "image messages can repeat older trade prices."
        ),
    }
    return frame, metadata
