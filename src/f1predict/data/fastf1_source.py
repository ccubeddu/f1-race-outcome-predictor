from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from f1predict.data.io import atomic_parquet_write


def _plain(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in frame.iterrows():
        rows.append({target: _plain(source.get(origin)) for origin, target in columns.items()})
    return pd.DataFrame(rows, columns=list(columns.values()))


LAP_COLUMNS = {
    "Driver": "driver_code",
    "DriverNumber": "driver_number",
    "LapNumber": "lap_number",
    "LapTime": "lap_time_seconds",
    "Sector1Time": "sector1_seconds",
    "Sector2Time": "sector2_seconds",
    "Sector3Time": "sector3_seconds",
    "PitOutTime": "pit_out_seconds",
    "PitInTime": "pit_in_seconds",
    "SpeedI1": "speed_i1_kph",
    "SpeedI2": "speed_i2_kph",
    "SpeedFL": "speed_finish_kph",
    "SpeedST": "speed_trap_kph",
    "IsPersonalBest": "is_personal_best",
    "Compound": "compound",
    "TyreLife": "tyre_life",
    "FreshTyre": "fresh_tyre",
    "Team": "team_name",
    "LapStartTime": "lap_start_seconds",
    "LapStartDate": "lap_start_date",
    "TrackStatus": "track_status",
    "Position": "position",
    "Deleted": "deleted",
    "DeletedReason": "deleted_reason",
    "FastF1Generated": "fastf1_generated",
    "IsAccurate": "is_accurate",
}

WEATHER_COLUMNS = {
    "Time": "session_time_seconds",
    "AirTemp": "air_temp_c",
    "Humidity": "humidity_pct",
    "Pressure": "pressure_mbar",
    "Rainfall": "rainfall",
    "TrackTemp": "track_temp_c",
    "WindDirection": "wind_direction_deg",
    "WindSpeed": "wind_speed_ms",
}

RESULT_COLUMNS = {
    "DriverNumber": "driver_number",
    "Abbreviation": "driver_code",
    "FullName": "driver_name",
    "TeamName": "team_name",
    "TeamColor": "team_color",
    "Position": "position",
    "ClassifiedPosition": "classified_position",
    "GridPosition": "grid_position",
    "Q1": "q1_seconds",
    "Q2": "q2_seconds",
    "Q3": "q3_seconds",
    "Time": "time_seconds",
    "Status": "status",
    "Points": "points",
}


class FastF1Source:
    """Downloads detailed sessions without telemetry and writes stable Parquet partitions."""

    def __init__(self, cache_dir: Path, output_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.configure(cache_dir=str(cache_dir))
        self.output_dir = output_dir

    def sync(
        self,
        start: int,
        end: int,
        sessions: Iterable[str],
        *,
        rounds: set[int] | None = None,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        session_codes = tuple(
            dict.fromkeys(code.strip().upper() for code in sessions if code.strip())
        )
        for season in range(start, end + 1):
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            season_rounds = [
                int(value)
                for value in schedule["RoundNumber"]
                if int(value) > 0 and (rounds is None or int(value) in rounds)
            ]
            for round_number in season_rounds:
                for session_code in session_codes:
                    partition = (
                        self.output_dir
                        / "laps"
                        / f"season={season}"
                        / f"round={round_number:02d}"
                        / f"session={session_code}"
                        / "data.parquet"
                    )
                    if partition.exists() and not refresh:
                        outcomes.append(
                            {
                                "season": season,
                                "round": round_number,
                                "session": session_code,
                                "status": "cached",
                            }
                        )
                        continue
                    try:
                        session = fastf1.get_session(season, round_number, session_code)
                        session.load(
                            laps=True,
                            telemetry=False,
                            weather=True,
                            messages=False,
                        )
                        context = {
                            "season": season,
                            "round": round_number,
                            "event_name": session.event.get("EventName"),
                            "session": session_code,
                            "session_name": session.name,
                            "session_date": _plain(session.date),
                        }
                        laps = _records(session.laps, LAP_COLUMNS).assign(**context)
                        weather = _records(session.weather_data, WEATHER_COLUMNS).assign(**context)
                        results = _records(session.results, RESULT_COLUMNS).assign(**context)
                        atomic_parquet_write(laps, partition)
                        atomic_parquet_write(
                            weather,
                            Path(str(partition).replace("/laps/", "/weather/")),
                        )
                        atomic_parquet_write(
                            results,
                            Path(str(partition).replace("/laps/", "/session_results/")),
                        )
                        outcomes.append(
                            {
                                "season": season,
                                "round": round_number,
                                "session": session_code,
                                "status": "downloaded",
                                "laps": len(laps),
                                "weather_samples": len(weather),
                            }
                        )
                    except Exception as exc:  # unavailable/cancelled/future sessions are expected
                        outcomes.append(
                            {
                                "season": season,
                                "round": round_number,
                                "session": session_code,
                                "status": "unavailable",
                                "detail": f"{type(exc).__name__}: {exc}",
                            }
                        )
        return outcomes
