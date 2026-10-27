from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from f1predict import __version__
from f1predict.data.io import atomic_json_write

Json = dict[str, Any]


def _number(value: Any, kind: type[int] | type[float] = float) -> int | float | None:
    if value in (None, "", "\\N"):
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def duration_seconds(value: Any) -> float | None:
    """Convert F1 time strings (SS.mmm, M:SS.mmm, H:MM:SS.mmm) to seconds."""
    if value in (None, "", "\\N"):
        return None
    text = str(value).lstrip("+")
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


class JolpicaClient:
    """Small cached client for the stable Ergast-compatible Jolpica API."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool = False,
        timeout_seconds: float = 30.0,
        minimum_interval_seconds: float = 0.25,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = "https://api.jolpi.ca/ergast/f1"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_request_at = 0.0
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": f"F1Predict/{__version__} httpx/{httpx.__version__}"},
            transport=transport,
        )

    def __enter__(self) -> JolpicaClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        identity = json.dumps([path.strip("/"), sorted(params.items())], separators=(",", ":"))
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        label = path.strip("/").replace("/", "__") or "root"
        return self.cache_dir / label / f"{digest}.json"

    def get(self, path: str, params: dict[str, Any] | None = None) -> Json:
        query = dict(params or {})
        cache_path = self._cache_path(path, query)
        if cache_path.exists() and not self.refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.base_url}/{path.strip('/')}"
        for attempt in range(5):
            wait = self.minimum_interval_seconds - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self._client.get(url, params=query)
            except httpx.TransportError:
                self._last_request_at = time.monotonic()
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            self._last_request_at = time.monotonic()
            if response.status_code == 404:
                payload: Json = {"MRData": {"limit": "0", "offset": "0", "total": "0"}}
                break
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            break
        else:  # pragma: no cover - loop always exits through break or exception
            raise RuntimeError(f"Unable to fetch {url}")

        atomic_json_write(cache_path, payload)
        atomic_json_write(
            cache_path.with_suffix(".meta.json"),
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                "url": str(response.url) if "response" in locals() else url,
                "sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
        )
        return payload

    def pages(self, path: str) -> Iterator[Json]:
        offset = 0
        while True:
            payload = self.get(path, {"limit": 100, "offset": offset})
            yield payload
            meta = payload.get("MRData", {})
            total = int(meta.get("total", 0))
            returned_limit = int(meta.get("limit", 100)) or 100
            offset += returned_limit
            if offset >= total:
                return


def _race_context(race: Json) -> Json:
    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})
    return {
        "season": _number(race.get("season"), int),
        "round": _number(race.get("round"), int),
        "race_name": race.get("raceName"),
        "race_date": race.get("date"),
        "race_time": race.get("time"),
        "circuit_id": circuit.get("circuitId"),
        "circuit_name": circuit.get("circuitName"),
        "locality": location.get("locality"),
        "country": location.get("country"),
        "latitude": _number(location.get("lat")),
        "longitude": _number(location.get("long")),
    }


def _driver_context(item: Json) -> Json:
    driver = item.get("Driver", {})
    constructor = item.get("Constructor", {})
    return {
        "driver_id": driver.get("driverId"),
        "driver_code": driver.get("code"),
        "driver_number": _number(driver.get("permanentNumber"), int),
        "given_name": driver.get("givenName"),
        "family_name": driver.get("familyName"),
        "driver_dob": driver.get("dateOfBirth"),
        "driver_nationality": driver.get("nationality"),
        "constructor_id": constructor.get("constructorId"),
        "constructor_name": constructor.get("name"),
        "constructor_nationality": constructor.get("nationality"),
    }


def _race_tables(pages: Iterable[Json]) -> Iterator[Json]:
    for payload in pages:
        table = payload.get("MRData", {}).get("RaceTable", {})
        yield from table.get("Races", [])


def parse_races(pages: Iterable[Json]) -> pd.DataFrame:
    rows: list[Json] = []
    for race in _race_tables(pages):
        row = _race_context(race)
        for source, prefix in (
            ("FirstPractice", "fp1"),
            ("SecondPractice", "fp2"),
            ("ThirdPractice", "fp3"),
            ("SprintQualifying", "sprint_quali"),
            ("Sprint", "sprint"),
            ("Qualifying", "qualifying"),
        ):
            session = race.get(source, {})
            row[f"{prefix}_date"] = session.get("date")
            row[f"{prefix}_time"] = session.get("time")
        rows.append(row)
    return pd.DataFrame(rows)


def parse_results(pages: Iterable[Json], *, collection: str = "Results") -> pd.DataFrame:
    rows: list[Json] = []
    for race in _race_tables(pages):
        base = _race_context(race)
        for item in race.get(collection, []):
            fastest = item.get("FastestLap", {})
            average_speed = fastest.get("AverageSpeed", {})
            rows.append(
                {
                    **base,
                    **_driver_context(item),
                    "car_number": _number(item.get("number"), int),
                    "grid_position": _number(item.get("grid"), int),
                    "finish_position": _number(item.get("position"), int),
                    "finish_position_text": item.get("positionText"),
                    "finish_order": _number(item.get("position"), int),
                    "points": _number(item.get("points")),
                    "laps_completed": _number(item.get("laps"), int),
                    "status": item.get("status"),
                    "race_time_seconds": duration_seconds(item.get("Time", {}).get("time")),
                    "fastest_lap_rank": _number(fastest.get("rank"), int),
                    "fastest_lap_number": _number(fastest.get("lap"), int),
                    "fastest_lap_seconds": duration_seconds(fastest.get("Time", {}).get("time")),
                    "average_speed_kph": _number(average_speed.get("speed")),
                }
            )
    return pd.DataFrame(rows)


def parse_qualifying(pages: Iterable[Json]) -> pd.DataFrame:
    rows: list[Json] = []
    for race in _race_tables(pages):
        base = _race_context(race)
        for item in race.get("QualifyingResults", []):
            rows.append(
                {
                    **base,
                    **_driver_context(item),
                    "car_number": _number(item.get("number"), int),
                    "quali_position": _number(item.get("position"), int),
                    "q1": item.get("Q1"),
                    "q2": item.get("Q2"),
                    "q3": item.get("Q3"),
                    "q1_seconds": duration_seconds(item.get("Q1")),
                    "q2_seconds": duration_seconds(item.get("Q2")),
                    "q3_seconds": duration_seconds(item.get("Q3")),
                }
            )
    return pd.DataFrame(rows)


def parse_pit_stops(pages: Iterable[Json]) -> pd.DataFrame:
    rows: list[Json] = []
    for race in _race_tables(pages):
        base = _race_context(race)
        for item in race.get("PitStops", []):
            rows.append(
                {
                    **base,
                    "driver_id": item.get("driverId"),
                    "stop": _number(item.get("stop"), int),
                    "lap": _number(item.get("lap"), int),
                    "local_time": item.get("time"),
                    "duration": item.get("duration"),
                    "duration_seconds": duration_seconds(item.get("duration")),
                }
            )
    return pd.DataFrame(rows)


def download_season(client: JolpicaClient, season: int) -> dict[str, pd.DataFrame]:
    """Download and normalize compact season-level data without lap timing bulk."""
    races = parse_races(client.pages(f"{season}.json"))
    pit_stop_frames = [
        parse_pit_stops(client.pages(f"{season}/{round_number}/pitstops.json"))
        for round_number in races.get("round", pd.Series(dtype="int64")).dropna().astype(int)
        if season >= 2011  # The Jolpica pit-stop endpoint starts in 2011.
    ]
    pit_stops = (
        pd.concat(pit_stop_frames, ignore_index=True)
        if any(not frame.empty for frame in pit_stop_frames)
        else pd.DataFrame()
    )
    return {
        "races": races,
        "results": parse_results(client.pages(f"{season}/results.json")),
        "qualifying": parse_qualifying(client.pages(f"{season}/qualifying.json")),
        "sprints": parse_results(client.pages(f"{season}/sprint.json"), collection="SprintResults"),
        "pit_stops": pit_stops,
    }
