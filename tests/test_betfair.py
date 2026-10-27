from __future__ import annotations

import bz2
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from f1predict.data.betfair import import_betfair_snapshot

MARKET = "1.12345"
DRIVERS = {"10": "verstappen", "20": "norris", "30": "leclerc"}
START = pd.Timestamp("2025-07-06T12:00:00Z")
CUTOFF = pd.Timestamp("2025-07-06T12:02:00Z")


def _definition() -> dict[str, Any]:
    return {
        "name": "Race Winner",
        "eventName": "British Grand Prix",
        "marketType": "WIN",
        "bettingType": "ODDS",
        "numberOfWinners": 1,
        "inPlay": False,
        "status": "OPEN",
        "complete": True,
        "marketTime": "2025-07-06T14:00:00Z",
        "numberOfActiveRunners": 3,
        "runners": [{"id": int(key), "status": "ACTIVE"} for key in DRIVERS],
    }


def _message(seconds: int = 0, **change: Any) -> dict[str, Any]:
    return {
        "op": "mcm",
        "pt": (START.value // 1_000_000) + seconds * 1000,
        "mc": [{"id": MARKET, **change}],
    }


def _image(seconds: int = 0) -> dict[str, Any]:
    return _message(
        seconds,
        img=True,
        marketDefinition=_definition(),
        rc=[{"id": 10, "ltp": 2.0}, {"id": 20, "ltp": 3.0}, {"id": 30, "ltp": 6.0}],
    )


def _stream(tmp_path: Path, messages: list[dict[str, Any]], *, compressed: bool = False) -> Path:
    content = "\n".join(json.dumps(message) for message in messages) + "\n"
    path = tmp_path / ("market.bz2" if compressed else "market.ndjson")
    if compressed:
        path.write_bytes(bz2.compress(content.encode()))
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_asof_reconstructs_deltas_without_future_prices_or_status(tmp_path: Path) -> None:
    future_definition = {**_definition(), "inPlay": True, "status": "CLOSED"}
    path = _stream(
        tmp_path,
        [
            _image(),
            _message(60, rc=[{"id": 10, "ltp": 1.8}, {"id": 20, "ltp": None}]),
            _message(120, rc=[{"id": 30, "ltp": 7.0}]),
            _message(180, marketDefinition=future_definition, rc=[{"id": 10, "ltp": 1.01}]),
        ],
    )
    frame, meta = import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)
    by_driver = frame.set_index("driver_id")
    assert by_driver.loc["verstappen", "decimal_odds"] == 1.8
    assert by_driver.loc["leclerc", "decimal_odds"] == 7.0
    assert by_driver.loc["norris", "quote_at"] == START.isoformat()
    assert by_driver.loc["verstappen", "quote_at"] == "2025-07-06T12:01:00+00:00"
    assert meta["asof"] == CUTOFF.isoformat()
    assert meta["cutoff"] == CUTOFF.isoformat()
    assert meta["applied_market_changes"] == 3
    assert meta["runner_count"] == 3
    assert meta["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["price_type"] == "last_traded"
    assert "not executable" in meta["caveat"]


def test_bz2_and_multiple_markets_use_only_explicit_market(tmp_path: Path) -> None:
    unrelated = _image(20)
    unrelated["mc"][0]["id"] = "1.9999"
    unrelated["mc"][0]["rc"] = [{"id": 999, "ltp": 1.01}]
    path = _stream(tmp_path, [_image(), unrelated], compressed=True)
    frame, meta = import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)
    assert frame["decimal_odds"].tolist() == [6.0, 3.0, 2.0]
    assert meta["asof"] == START.isoformat()
    with pytest.raises(ValueError, match="No marketDefinition"):
        import_betfair_snapshot(path, DRIVERS, "1.404", CUTOFF)


def test_future_message_does_not_hide_later_eligible_update_in_source(tmp_path: Path) -> None:
    path = _stream(
        tmp_path,
        [
            _image(),
            _message(200, rc=[{"id": 10, "ltp": 1.01}]),
            _message(60, rc=[{"id": 20, "ltp": 3.5}]),
        ],
    )
    frame, _ = import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)
    assert frame.set_index("driver_id").loc["norris", "decimal_odds"] == 3.5


def test_img_resets_prices_instead_of_carrying_previous_quotes(tmp_path: Path) -> None:
    reset = _image(60)
    reset["mc"][0]["rc"] = [{"id": 10, "ltp": 1.8}]
    path = _stream(tmp_path, [_image(), reset])
    with pytest.raises(ValueError, match="price for every runner"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


def test_img_resets_definition_instead_of_carrying_previous_fields(tmp_path: Path) -> None:
    path = _stream(tmp_path, [_image(), _message(60, img=True, rc=[{"id": 10, "ltp": 1.8}])])
    with pytest.raises(ValueError, match="No marketDefinition"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


def test_stale_runner_price_is_not_refreshed_by_market_activity(tmp_path: Path) -> None:
    path = _stream(
        tmp_path,
        [_image(), _message(120, rc=[{"id": 10, "ltp": 1.8}, {"id": 20, "ltp": None}])],
    )
    with pytest.raises(ValueError, match="Stale last-traded price for norris"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF, max_age_seconds=119)
    # The maximum age is inclusive and measured against cutoff, not asof.
    import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF, max_age_seconds=120)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("inPlay", True, "pre-race"),
        ("inPlay", None, "pre-race"),
        ("status", "SUSPENDED", "OPEN"),
        ("status", "CLOSED", "OPEN"),
        ("marketType", "PLACE", "WIN/ODDS"),
        ("marketType", None, "WIN/ODDS"),
        ("bettingType", "LINE", "WIN/ODDS"),
        ("numberOfWinners", 3, "numberOfWinners=1"),
        ("numberOfWinners", None, "numberOfWinners=1"),
        ("numberOfWinners", True, "numberOfWinners=1"),
        ("complete", False, "complete grid"),
        ("numberOfActiveRunners", 2, "numberOfActiveRunners"),
        ("runners", None, "all runners"),
        ("marketTime", None, "marketTime"),
        ("marketTime", "2025-07-06T14:00:00", "marketTime"),
    ],
)
def test_rejects_unsafe_or_missing_definition_fields(
    tmp_path: Path, field: str, value: Any, error: str
) -> None:
    image = _image()
    image["mc"][0]["marketDefinition"][field] = value
    path = _stream(tmp_path, [image])
    with pytest.raises(ValueError, match=error):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


@pytest.mark.parametrize("status", ["REMOVED", "WINNER", "LOSER", None])
def test_rejects_any_nonactive_runner(tmp_path: Path, status: str | None) -> None:
    image = _image()
    image["mc"][0]["marketDefinition"]["runners"][0]["status"] = status
    path = _stream(tmp_path, [image])
    with pytest.raises(ValueError, match="must be ACTIVE"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


def test_requires_exact_explicit_selection_mapping(tmp_path: Path) -> None:
    path = _stream(tmp_path, [_image()])
    with pytest.raises(ValueError, match="Unmapped selection"):
        import_betfair_snapshot(path, {"10": "verstappen", "20": "norris"}, MARKET, CUTOFF)
    with pytest.raises(ValueError, match="Incomplete grid/driver mapping"):
        import_betfair_snapshot(path, {**DRIVERS, "40": "hamilton"}, MARKET, CUTOFF)
    with pytest.raises(ValueError, match="distinct"):
        import_betfair_snapshot(path, {**DRIVERS, "30": "norris"}, MARKET, CUTOFF)
    image = _image()
    image["mc"][0]["rc"] = []
    path = _stream(tmp_path, [image])
    with pytest.raises(ValueError, match="price for every runner"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


def test_cutoff_must_be_timezone_aware_and_before_market_start(tmp_path: Path) -> None:
    path = _stream(tmp_path, [_image()])
    with pytest.raises(ValueError, match="timezone-aware"):
        import_betfair_snapshot(path, DRIVERS, MARKET, "2025-07-06T12:02:00")
    with pytest.raises(ValueError, match="precede"):
        import_betfair_snapshot(path, DRIVERS, MARKET, "2025-07-06T14:00:00Z")
    frame, meta = import_betfair_snapshot(path, DRIVERS, MARKET, "2025-07-06T08:02:00-04:00")
    assert meta["cutoff"] == CUTOFF.isoformat()
    assert len(frame) == 3


def test_rejects_timestamp_regressions_and_unsupported_segments(tmp_path: Path) -> None:
    path = _stream(tmp_path, [_image(), _message(90, rc=[]), _message(60, rc=[])])
    with pytest.raises(ValueError, match="timestamps regress"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)
    segmented = _image()
    segmented["segmentType"] = "SEG_START"
    path = _stream(tmp_path, [segmented])
    with pytest.raises(ValueError, match="Segmented"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


def test_definition_update_replaces_whole_object(tmp_path: Path) -> None:
    # A damaged partial definition must not inherit safety fields from the past.
    path = _stream(tmp_path, [_image(), _message(60, marketDefinition={"status": "OPEN"})])
    with pytest.raises(ValueError, match="WIN/ODDS"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)


@pytest.mark.parametrize("market_type", ["WINNER", "OUTRIGHT_WINNER"])
def test_other_winner_codes_require_an_explicit_exact_match(
    tmp_path: Path, market_type: str
) -> None:
    image = _image()
    image["mc"][0]["marketDefinition"]["marketType"] = market_type
    path = _stream(tmp_path, [image])
    with pytest.raises(ValueError, match="WIN/ODDS"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)
    frame, metadata = import_betfair_snapshot(
        path, DRIVERS, MARKET, CUTOFF, expected_market_type=market_type
    )
    assert len(frame) == 3
    assert metadata["market_type"] == market_type
    assert metadata["expected_market_type"] == market_type


def test_explicit_market_type_cannot_enable_a_place_market(tmp_path: Path) -> None:
    image = _image()
    image["mc"][0]["marketDefinition"]["marketType"] = "PLACE"
    path = _stream(tmp_path, [image])
    with pytest.raises(ValueError, match="explicit winner code"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF, expected_market_type="PLACE")


@pytest.mark.parametrize("price", [1.0, 0, -2, 1001, float("nan"), float("inf"), True, "2.0"])
def test_rejects_invalid_last_traded_prices(tmp_path: Path, price: Any) -> None:
    image = deepcopy(_image())
    image["mc"][0]["rc"][0]["ltp"] = price
    path = _stream(tmp_path, [image])
    with pytest.raises(ValueError, match="Invalid decimal"):
        import_betfair_snapshot(path, DRIVERS, MARKET, CUTOFF)
