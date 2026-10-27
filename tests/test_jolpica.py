from __future__ import annotations

import httpx
import pytest

from f1predict.data.jolpica import JolpicaClient, duration_seconds, parse_qualifying, parse_results


def payload(race: dict) -> list[dict]:
    return [{"MRData": {"RaceTable": {"Races": [race]}}}]


BASE_RACE = {
    "season": "2024",
    "round": "1",
    "raceName": "Bahrain Grand Prix",
    "date": "2024-03-02",
    "Circuit": {
        "circuitId": "bahrain",
        "circuitName": "Bahrain International Circuit",
        "Location": {"lat": "26.0325", "long": "50.5106", "country": "Bahrain"},
    },
}

DRIVER = {
    "Driver": {
        "driverId": "example",
        "code": "EXA",
        "givenName": "Example",
        "familyName": "Driver",
    },
    "Constructor": {"constructorId": "example_team", "name": "Example Team"},
    "number": "1",
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("23.456", 23.456), ("1:23.456", 83.456), ("1:02:03.400", 3723.4), (None, None)],
)
def test_duration_seconds(raw: str | None, expected: float | None) -> None:
    assert duration_seconds(raw) == expected


def test_parse_results_preserves_official_order_and_status() -> None:
    race = {
        **BASE_RACE,
        "Results": [
            {
                **DRIVER,
                "grid": "2",
                "position": "1",
                "positionText": "1",
                "points": "25",
                "laps": "57",
                "status": "Finished",
                "Time": {"time": "1:31:44.742"},
            }
        ],
    }
    frame = parse_results(payload(race))
    assert frame.loc[0, "driver_id"] == "example"
    assert frame.loc[0, "finish_position"] == 1
    assert frame.loc[0, "grid_position"] == 2
    assert frame.loc[0, "status"] == "Finished"


def test_parse_qualifying_converts_session_times() -> None:
    race = {
        **BASE_RACE,
        "QualifyingResults": [
            {**DRIVER, "position": "1", "Q1": "1:30.000", "Q2": "1:29.000", "Q3": "1:28.000"}
        ],
    }
    frame = parse_qualifying(payload(race))
    assert frame.loc[0, "quali_position"] == 1
    assert frame.loc[0, "q3_seconds"] == 88.0


def test_transient_disconnect_retries_and_then_reuses_cached_response(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def handler(request):
        calls.append(request.url)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        return httpx.Response(200, json={"MRData": {"total": "19"}})

    monkeypatch.setattr("f1predict.data.jolpica.time.sleep", lambda seconds: None)
    with JolpicaClient(
        tmp_path, minimum_interval_seconds=0, transport=httpx.MockTransport(handler)
    ) as client:
        first = client.get("2010.json")
        assert client.get("2010.json") == first
    assert len(calls) == 2
