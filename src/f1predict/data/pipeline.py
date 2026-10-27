from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from f1predict.config import DataPaths
from f1predict.data.contracts import CONTRACTS, validate_table
from f1predict.data.f1db import f1db_practice_data
from f1predict.data.fastf1_source import FastF1Source
from f1predict.data.io import atomic_json_write, atomic_parquet_write, read_partitioned_table
from f1predict.data.jolpica import JolpicaClient, download_season
from f1predict.data.openf1 import OpenF1Source


def _catalog(paths: DataPaths) -> dict[str, Any]:
    if not paths.catalog.exists():
        return {"schema_version": 1, "sources": {}}
    return json.loads(paths.catalog.read_text(encoding="utf-8"))


def sync_jolpica(
    paths: DataPaths,
    start: int,
    end: int,
    *,
    refresh: bool = False,
    progress: Callable[[int, dict[str, int]], None] | None = None,
) -> dict[int, dict[str, int]]:
    paths.create()
    summary: dict[int, dict[str, int]] = {}
    with JolpicaClient(paths.raw_jolpica, refresh=refresh) as client:
        for season in range(start, end + 1):
            frames = download_season(client, season)
            season_summary: dict[str, int] = {}
            for name, frame in frames.items():
                if frame.empty and not len(frame.columns):
                    frame = frame.reindex(columns=CONTRACTS[name].required)
                validate_table(name, frame)
                target = paths.processed_jolpica / name / f"season={season}" / "data.parquet"
                atomic_parquet_write(frame, target)
                season_summary[name] = len(frame)
            summary[season] = season_summary
            if progress:
                progress(season, season_summary)

    catalog = _catalog(paths)
    previous = catalog["sources"].get("jolpica", {}).get("rows", {})
    all_rows = {str(year): counts for year, counts in previous.items()}
    all_rows.update({str(year): counts for year, counts in summary.items()})
    catalog["sources"]["jolpica"] = {
        "base_url": "https://api.jolpi.ca/ergast/f1",
        "refreshed_at": datetime.now(UTC).isoformat(),
        "season_start": min(int(year) for year in all_rows),
        "season_end": max(int(year) for year in all_rows),
        "rows": all_rows,
        "last_sync": {"season_start": start, "season_end": end},
    }
    atomic_json_write(paths.catalog, catalog)
    return summary


def sync_fastf1(
    paths: DataPaths,
    start: int,
    end: int,
    sessions: list[str],
    *,
    rounds: set[int] | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    paths.create()
    source = FastF1Source(paths.fastf1_cache, paths.processed_fastf1)
    outcomes = source.sync(start, end, sessions, rounds=rounds, refresh=refresh)
    catalog = _catalog(paths)
    catalog["sources"]["fastf1"] = {
        "refreshed_at": datetime.now(UTC).isoformat(),
        "season_start": start,
        "season_end": end,
        "sessions": sessions,
        "rounds": sorted(rounds) if rounds else "all",
        "downloaded": sum(row["status"] == "downloaded" for row in outcomes),
        "cached": sum(row["status"] == "cached" for row in outcomes),
        "unavailable": sum(row["status"] == "unavailable" for row in outcomes),
    }
    atomic_json_write(paths.catalog, catalog)
    return outcomes


def sync_f1db(
    paths: DataPaths,
    start: int,
    end: int,
    *,
    refresh: bool = False,
) -> dict[int, int]:
    paths.create()
    frame = f1db_practice_data(paths.raw_f1db, start, end, refresh=refresh)
    summary: dict[int, int] = {}
    for season, partition in frame.groupby("season", observed=True):
        partition = partition.reset_index(drop=True)
        validate_table("f1db_practice_results", partition)
        target = (
            paths.processed_f1db / "practice_results" / f"season={int(season)}" / "data.parquet"
        )
        atomic_parquet_write(partition, target)
        summary[int(season)] = len(partition)
    catalog = _catalog(paths)
    previous = catalog["sources"].get("f1db", {}).get("practice_rows", {})
    all_rows = {str(year): count for year, count in previous.items()}
    all_rows.update({str(year): count for year, count in summary.items()})
    catalog["sources"]["f1db"] = {
        "source_url": "https://github.com/f1db/f1db/releases/latest",
        "license": "CC-BY-4.0",
        "refreshed_at": datetime.now(UTC).isoformat(),
        "season_start": min((int(year) for year in all_rows), default=start),
        "season_end": max((int(year) for year in all_rows), default=end),
        "practice_rows": all_rows,
        "last_sync": {"season_start": start, "season_end": end},
    }
    atomic_json_write(paths.catalog, catalog)
    return summary


def sync_openf1(
    paths: DataPaths,
    start: int,
    end: int,
    *,
    rounds: set[int] | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    paths.create()
    races = read_partitioned_table(paths.processed_jolpica, "races")
    if races.empty:
        raise RuntimeError("Jolpica races are required first; run `f1predict data sync`.")
    source = OpenF1Source(paths.raw_openf1, paths.processed_openf1, refresh=refresh)
    outcomes = source.sync(races, start, end, rounds=rounds)
    catalog = _catalog(paths)
    catalog["sources"]["openf1"] = {
        "base_url": "https://api.openf1.org/v1",
        "refreshed_at": datetime.now(UTC).isoformat(),
        "season_start": max(2023, start),
        "season_end": end,
        "rounds": sorted(rounds) if rounds else "all",
        "downloaded": sum(row["status"] == "downloaded" for row in outcomes),
        "cached": sum(row["status"] == "cached" for row in outcomes),
        "unavailable": sum(row["status"] == "unavailable" for row in outcomes),
        "lap_rows": sum(int(row.get("laps", 0)) for row in outcomes),
    }
    atomic_json_write(paths.catalog, catalog)
    return outcomes
