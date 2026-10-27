from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from f1predict import __version__
from f1predict.data.io import atomic_json_write

F1DB_URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"


def download_f1db_archive(raw_dir: Path, *, refresh: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "f1db-csv.zip"
    if target.exists() and not refresh:
        return target
    with (
        httpx.Client(
            follow_redirects=True,
            timeout=120,
            headers={"User-Agent": f"F1Predict/{__version__}"},
        ) as client,
        client.stream("GET", F1DB_URL) as response,
    ):
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False, suffix=".zip") as handle:
            digest = hashlib.sha256()
            for chunk in response.iter_bytes():
                handle.write(chunk)
                digest.update(chunk)
            temporary = Path(handle.name)
        final_url = str(response.url)
    os.replace(temporary, target)
    atomic_json_write(
        raw_dir / "f1db-csv.meta.json",
        {
            "downloaded_at": datetime.now(UTC).isoformat(),
            "source_url": F1DB_URL,
            "resolved_url": final_url,
            "sha256": digest.hexdigest(),
            "license": "CC-BY-4.0",
        },
    )
    return target


def _read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)), low_memory=False)


def parse_practice_archive(archive_path: Path) -> pd.DataFrame:
    """Normalize FP1/FP2/FP3 classifications from an F1DB release."""
    with zipfile.ZipFile(archive_path) as archive:
        drivers = _read_csv(archive, "f1db-drivers.csv")[["id", "fullName", "abbreviation"]].rename(
            columns={
                "id": "driverId",
                "fullName": "driver_name",
                "abbreviation": "driver_code",
            }
        )
        races = _read_csv(archive, "f1db-races.csv")
        schedule_columns = [
            "id",
            "freePractice1Date",
            "freePractice1Time",
            "freePractice2Date",
            "freePractice2Time",
            "freePractice3Date",
            "freePractice3Time",
            "qualifyingDate",
            "qualifyingTime",
        ]
        schedules = races[schedule_columns].rename(columns={"id": "raceId"})
        frames: list[pd.DataFrame] = []
        for number in (1, 2, 3):
            frame = _read_csv(archive, f"f1db-races-free-practice-{number}-results.csv")
            frame["session"] = f"FP{number}"
            frame = frame.merge(drivers, how="left", on="driverId").merge(
                schedules, how="left", on="raceId"
            )
            frame["session_date"] = frame[f"freePractice{number}Date"]
            frame["session_time"] = frame[f"freePractice{number}Time"]
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    normalized = pd.DataFrame(
        {
            "season": pd.to_numeric(combined["year"], errors="coerce").astype("Int64"),
            "round": pd.to_numeric(combined["round"], errors="coerce").astype("Int64"),
            "race_id": combined["raceId"],
            "session": combined["session"],
            "session_date": combined["session_date"].astype("string"),
            "session_time": combined["session_time"].astype("string"),
            "qualifying_date": combined["qualifyingDate"].astype("string"),
            "qualifying_time": combined["qualifyingTime"].astype("string"),
            "position": pd.to_numeric(combined["positionDisplayOrder"], errors="coerce").astype(
                "Int64"
            ),
            "position_text": combined["positionText"].astype("string"),
            "driver_number": pd.to_numeric(combined["driverNumber"], errors="coerce").astype(
                "Int64"
            ),
            "driver_id": combined["driverId"].astype("string"),
            "driver_code": combined["driver_code"].astype("string"),
            "driver_name": combined["driver_name"].astype("string"),
            "constructor_id": combined["constructorId"].astype("string"),
            "engine_manufacturer_id": combined["engineManufacturerId"].astype("string"),
            "tyre_manufacturer_id": combined["tyreManufacturerId"].astype("string"),
            "best_lap": combined["time"].astype("string"),
            "best_lap_seconds": pd.to_numeric(combined["timeMillis"], errors="coerce") / 1000,
            "gap": combined["gap"].astype("string"),
            "gap_seconds": pd.to_numeric(combined["gapMillis"], errors="coerce") / 1000,
            "laps_completed": pd.to_numeric(combined["laps"], errors="coerce").astype("Int64"),
            "source": "f1db",
            "availability_rule": "available_by_qualifying_start",
        }
    )
    return normalized.sort_values(["season", "round", "session", "position"], ignore_index=True)


def f1db_practice_data(
    raw_dir: Path,
    start: int,
    end: int,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    archive = download_f1db_archive(raw_dir, refresh=refresh)
    frame = parse_practice_archive(archive)
    return frame.loc[frame["season"].between(start, end)].reset_index(drop=True)
