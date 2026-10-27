from __future__ import annotations

import httpx
import pandas as pd
import pytest

from f1predict.data.openf1 import OpenF1Client, OpenF1Source, normalize_sessions


def test_normalize_sessions_maps_practice_to_race_round() -> None:
    races = pd.DataFrame([{"season": 2024, "round": 1, "race_date": "2024-03-02"}])
    sessions = [
        {
            "year": 2024,
            "meeting_key": 10,
            "session_key": 11,
            "session_name": "Practice 1",
            "session_type": "Practice",
            "date_start": "2024-02-29T11:30:00+00:00",
            "date_end": "2024-02-29T12:30:00+00:00",
            "is_cancelled": False,
            "circuit_key": 1,
            "circuit_short_name": "Sakhir",
            "country_name": "Bahrain",
            "location": "Sakhir",
        },
        {
            "year": 2024,
            "meeting_key": 9,
            "session_key": 8,
            "session_name": "Day 1",
            "session_type": "Practice",
            "date_start": "2024-02-21T07:00:00+00:00",
            "date_end": "2024-02-21T15:00:00+00:00",
            "is_cancelled": False,
        },
    ]
    frame = normalize_sessions(sessions, races)
    assert len(frame) == 1
    assert frame.loc[0, "round"] == 1
    assert frame.loc[0, "session"] == "FP1"


def test_client_retries_transport_and_reuses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("f1predict.data.openf1.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("transient disconnect")
        return httpx.Response(200, json=[{"session_key": 10}])

    with OpenF1Client(tmp_path, transport=httpx.MockTransport(handler)) as client:
        assert client.minimum_interval_seconds >= 2
        assert client.get("sessions", {"year": 2024}) == [{"session_key": 10}]
        assert client.get("sessions", {"year": 2024}) == [{"session_key": 10}]
    assert len(calls) == 2


def test_bulk_keeps_only_practice_and_matches_per_session_ingestion(tmp_path, monkeypatch) -> None:
    races = pd.DataFrame([dict(season=2024, round=1, race_date="2024-03-03")])
    practice = dict(
        year=2024,
        meeting_key=1,
        session_key=10,
        session_name="Practice 1",
        date_start="2024-03-01T10:00:00Z",
        date_end="2024-03-01T11:00:00Z",
    )
    qualifying = dict(practice, session_key=11, session_name="Qualifying")
    payloads = {
        "sessions": [practice, qualifying],
        "drivers": [
            dict(session_key=10, driver_number=1, name_acronym="VER", full_name="Max Verstappen")
        ],
        "session_result": [dict(session_key=k, driver_number=1, position=1) for k in (10, 11)],
        "laps": [
            dict(session_key=k, driver_number=1, lap_number=2, lap_duration=90) for k in (10, 11)
        ],
        "stints": [dict(session_key=10, driver_number=1, stint_number=1)],
        "weather": [dict(session_key=10, date="2024-03-01T10:10:00Z", air_temperature=20)],
    }
    calls = []

    def get(self, endpoint, params):
        calls.append((endpoint, params))
        if "session_key" in params:
            return [
                item for item in payloads[endpoint] if item["session_key"] == params["session_key"]
            ]
        return payloads[endpoint]

    monkeypatch.setattr(OpenF1Client, "get", get)
    bulk = OpenF1Source(tmp_path / "raw", tmp_path / "bulk")
    events = []
    result = bulk.sync(races, 2024, 2024, bulk=True, progress=events.append)
    assert result[0]["status"] == "downloaded"
    assert len(calls) == 6  # sessions + 5 bounded requests, not requests per practice
    assert calls[1][1] == {"session_key>": 9, "session_key<": 11}
    assert len([e for e in events if e["status"] == "bulk_endpoint"]) == 5
    single = OpenF1Source(tmp_path / "raw", tmp_path / "single", workers=1)
    assert single.sync(races, 2024, 2024)[0]["status"] == "downloaded"
    for table in ("practice_results", "practice_laps", "practice_stints", "practice_weather"):
        path = f"{table}/season=2024/round=01/session=FP1/data.parquet"
        actual = pd.read_parquet(tmp_path / "bulk" / path)
        pd.testing.assert_frame_equal(actual, pd.read_parquet(tmp_path / "single" / path))
        assert set(actual["session_key"]) == {10}


@pytest.mark.parametrize(
    "cancelled,end", [(True, "2024-03-01T11:00:00Z"), (False, "2099-03-01T11:00:00Z")]
)
def test_cancelled_or_incomplete_sessions_excluded(cancelled, end) -> None:
    races = pd.DataFrame([dict(season=2024, round=1, race_date="2024-03-03")])
    item = dict(
        year=2024,
        meeting_key=1,
        session_key=10,
        session_name="Practice 1",
        date_end=end,
        is_cancelled=cancelled,
    )
    assert normalize_sessions([item], races).empty
