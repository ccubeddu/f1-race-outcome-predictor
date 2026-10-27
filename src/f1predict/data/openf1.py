from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
import pandas as pd

from f1predict import __version__
from f1predict.data.contracts import validate_table
from f1predict.data.io import atomic_json_write, atomic_parquet_write

Json = dict[str, Any]


class OpenF1Client:
    """Cached client for free historical OpenF1 data."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool = False,
        minimum_interval_seconds: float = 0.15,
    ) -> None:
        self.base_url = "https://api.openf1.org/v1"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_request_at = 0.0
        self._pace_lock = Lock()
        self._client = httpx.Client(
            timeout=90,
            follow_redirects=True,
            headers={"User-Agent": f"F1Predict/{__version__} httpx/{httpx.__version__}"},
        )

    def __enter__(self) -> OpenF1Client:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        identity = json.dumps([endpoint, sorted(params.items())], separators=(",", ":"))
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return self.cache_dir / endpoint / f"{digest}.json"

    def get(self, endpoint: str, params: dict[str, Any]) -> list[Json]:
        path = self._cache_path(endpoint, params)
        if path.exists() and not self.refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(5):
            with self._pace_lock:
                wait = self.minimum_interval_seconds - (time.monotonic() - self._last_request_at)
                if wait > 0:
                    time.sleep(wait)
                self._last_request_at = time.monotonic()
            response = self._client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                received = type(payload).__name__
                raise ValueError(f"OpenF1 {endpoint} returned {received}, expected list")
            break
        else:  # pragma: no cover
            raise RuntimeError(f"Unable to fetch {url}")
        atomic_json_write(path, payload)
        atomic_json_write(
            path.with_suffix(".meta.json"),
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                "url": str(response.url),
                "sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
        )
        return payload


def _resolve_round(session: Json, races: pd.DataFrame) -> int | None:
    session_end = pd.to_datetime(session.get("date_end"), utc=True, errors="coerce")
    if pd.isna(session_end):
        return None
    candidates = races.loc[races["season"].eq(int(session["year"])), ["round", "race_date"]].copy()
    candidates["race_date"] = pd.to_datetime(candidates["race_date"], utc=True, errors="coerce")
    candidates["days_after_session"] = (
        candidates["race_date"].dt.normalize() - session_end.normalize()
    ).dt.days
    candidates = candidates.loc[candidates["days_after_session"].between(0, 5)]
    if candidates.empty:
        return None
    return int(candidates.sort_values("days_after_session").iloc[0]["round"])


def normalize_sessions(records: list[Json], races: pd.DataFrame) -> pd.DataFrame:
    rows: list[Json] = []
    now = pd.Timestamp.now(tz="UTC")
    for item in records:
        if item.get("session_name") not in {"Practice 1", "Practice 2", "Practice 3"}:
            continue
        if item.get("is_cancelled"):
            continue
        date_end = pd.to_datetime(item.get("date_end"), utc=True, errors="coerce")
        if pd.isna(date_end) or date_end > now:
            continue
        round_number = _resolve_round(item, races)
        if round_number is None:
            continue
        rows.append(
            {
                "season": int(item["year"]),
                "round": round_number,
                "meeting_key": int(item["meeting_key"]),
                "session_key": int(item["session_key"]),
                "session_name": item["session_name"],
                "session": item["session_name"].replace("Practice ", "FP"),
                "session_type": item.get("session_type"),
                "date_start": item.get("date_start"),
                "date_end": item.get("date_end"),
                "circuit_key": item.get("circuit_key"),
                "circuit_short_name": item.get("circuit_short_name"),
                "country_name": item.get("country_name"),
                "location": item.get("location"),
                "source": "openf1",
            }
        )
    return pd.DataFrame(rows)


def _frame(records: list[Json], context: Json) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for key, value in context.items():
        frame[key] = value
    return frame


def _with_driver_details(frame: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or drivers.empty or "driver_number" not in frame:
        return frame
    columns = [
        column
        for column in (
            "driver_number",
            "name_acronym",
            "full_name",
            "team_name",
            "team_colour",
        )
        if column in drivers
    ]
    details = drivers[columns].drop_duplicates("driver_number", keep="last")
    return frame.merge(details, how="left", on="driver_number")


class OpenF1Source:
    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        *,
        refresh: bool = False,
        workers: int = 2,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.refresh = refresh
        self.workers = workers

    def _sync_session(self, client: OpenF1Client, context: Json) -> Json:
        session_key = int(context["session_key"])
        season = int(context["season"])
        partition = (
            f"season={season}/round={int(context['round']):02d}/session={context['session']}"
        )
        targets = {
            name: self.output_dir / name / partition / "data.parquet"
            for name in (
                "practice_results",
                "practice_laps",
                "practice_stints",
                "practice_weather",
            )
        }
        if all(path.exists() for path in targets.values()) and not self.refresh:
            return {**context, "status": "cached"}
        try:
            drivers = pd.DataFrame(client.get("drivers", {"session_key": session_key}))
            request_context = {
                key: context[key]
                for key in (
                    "season",
                    "round",
                    "meeting_key",
                    "session_key",
                    "session",
                    "date_end",
                )
            }
            tables = {
                "practice_results": _with_driver_details(
                    _frame(
                        client.get("session_result", {"session_key": session_key}),
                        request_context,
                    ),
                    drivers,
                ),
                "practice_laps": _with_driver_details(
                    _frame(
                        client.get("laps", {"session_key": session_key}),
                        request_context,
                    ),
                    drivers,
                ),
                "practice_stints": _with_driver_details(
                    _frame(
                        client.get("stints", {"session_key": session_key}),
                        request_context,
                    ),
                    drivers,
                ),
                "practice_weather": _frame(
                    client.get("weather", {"session_key": session_key}),
                    request_context,
                ),
            }
            keys = {
                "practice_results": ["session_key", "driver_number"],
                "practice_laps": ["session_key", "driver_number", "lap_number"],
                "practice_stints": ["session_key", "driver_number", "stint_number"],
                "practice_weather": ["session_key", "date"],
            }
            for name, frame in tables.items():
                if not frame.empty:
                    frame = frame.drop_duplicates(keys[name], keep="last")
                validate_table(f"openf1_{name}", frame)
                atomic_parquet_write(frame, targets[name])
            return {
                **context,
                "status": "downloaded",
                "laps": len(tables["practice_laps"]),
                "weather_samples": len(tables["practice_weather"]),
            }
        except Exception as exc:
            return {
                **context,
                "status": "unavailable",
                "detail": f"{type(exc).__name__}: {exc}",
            }

    def sync(
        self,
        races: pd.DataFrame,
        start: int,
        end: int,
        *,
        rounds: set[int] | None = None,
    ) -> list[Json]:
        outcomes: list[Json] = []
        with OpenF1Client(self.raw_dir, refresh=self.refresh) as client:
            for season in range(max(2023, start), end + 1):
                sessions = normalize_sessions(client.get("sessions", {"year": season}), races)
                if rounds:
                    sessions = sessions.loc[sessions["round"].isin(rounds)]
                if sessions.empty:
                    continue
                requested_sessions = sessions.copy()
                session_target = (
                    self.output_dir / "practice_sessions" / f"season={season}" / "data.parquet"
                )
                if session_target.exists() and rounds:
                    existing = pd.read_parquet(session_target)
                    sessions = pd.concat([existing, sessions], ignore_index=True).drop_duplicates(
                        "session_key", keep="last"
                    )
                atomic_parquet_write(sessions, session_target)
                contexts = requested_sessions.to_dict("records")
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    outcomes.extend(
                        executor.map(lambda item: self._sync_session(client, item), contexts)
                    )
        return outcomes
