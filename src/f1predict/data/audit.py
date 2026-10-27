from __future__ import annotations

from typing import Any

import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.contracts import CONTRACTS
from f1predict.data.io import read_parquet_files, read_partitioned_table


def _integer(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def build_audit(paths: DataPaths) -> dict[str, Any]:
    report: dict[str, Any] = {
        "tables": {},
        "f1db": {},
        "openf1": {},
        "fastf1": {},
        "coverage": {},
        "warnings": [],
    }
    loaded: dict[str, pd.DataFrame] = {}
    core_tables = ("races", "results", "qualifying", "sprints", "pit_stops")
    for name in core_tables:
        contract = CONTRACTS[name]
        frame = read_partitioned_table(paths.processed_jolpica, name)
        loaded[name] = frame
        if frame.empty:
            report["tables"][name] = {"rows": 0, "status": "missing"}
            report["warnings"].append(f"{name}: no rows found")
            continue
        duplicates = int(frame.duplicated(list(contract.key)).sum())
        required_nulls = {
            column: int(frame[column].isna().sum())
            for column in contract.required
            if column in frame.columns and frame[column].isna().any()
        }
        report["tables"][name] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "season_min": _integer(frame["season"].min()),
            "season_max": _integer(frame["season"].max()),
            "duplicate_keys": duplicates,
            "required_nulls": required_nulls,
            "status": "ok" if not duplicates and not required_nulls else "warning",
        }

    f1db_practice = read_partitioned_table(paths.processed_f1db, "practice_results")
    if not f1db_practice.empty:
        contract = CONTRACTS["f1db_practice_results"]
        report["f1db"] = {
            "practice_rows": len(f1db_practice),
            "season_min": _integer(f1db_practice["season"].min()),
            "season_max": _integer(f1db_practice["season"].max()),
            "sessions": f1db_practice.groupby("session").size().to_dict(),
            "timed_lap_rate": float(f1db_practice["best_lap_seconds"].notna().mean()),
            "duplicate_keys": int(f1db_practice.duplicated(list(contract.key)).sum()),
        }
    else:
        report["f1db"] = {"practice_rows": 0, "status": "missing"}

    openf1_patterns = {
        "practice_sessions": "season=*/data.parquet",
        "practice_results": "season=*/round=*/session=*/data.parquet",
        "practice_laps": "season=*/round=*/session=*/data.parquet",
        "practice_stints": "season=*/round=*/session=*/data.parquet",
        "practice_weather": "season=*/round=*/session=*/data.parquet",
    }
    openf1_frames: dict[str, pd.DataFrame] = {}
    for name, pattern in openf1_patterns.items():
        files = sorted((paths.processed_openf1 / name).glob(pattern))
        frame = read_parquet_files(files)
        openf1_frames[name] = frame
        report["openf1"][name] = {"partitions": len(files), "rows": len(frame)}
    laps = openf1_frames["practice_laps"]
    if not laps.empty:
        report["openf1"]["practice_laps"]["timed_laps"] = int(laps["lap_duration"].notna().sum())
        report["openf1"]["practice_laps"]["sessions"] = int(laps["session_key"].nunique())
        report["openf1"]["practice_laps"]["drivers"] = int(
            laps[["session_key", "driver_number"]].drop_duplicates().shape[0]
        )

    results = loaded["results"]
    qualifying = loaded["qualifying"]
    races = loaded["races"]
    if not results.empty:
        grid_sizes = results.groupby(["season", "round"]).size()
        report["coverage"]["result_grid_size"] = {
            "minimum": int(grid_sizes.min()),
            "median": float(grid_sizes.median()),
            "maximum": int(grid_sizes.max()),
        }
        report["coverage"]["non_finished_status_examples"] = (
            results.loc[results["status"] != "Finished", "status"].value_counts().head(10).to_dict()
        )
    if not results.empty and not qualifying.empty:
        result_keys = results[["season", "round", "driver_id"]]
        quali_keys = qualifying[["season", "round", "driver_id"]].drop_duplicates()
        joined = result_keys.merge(quali_keys.assign(has_quali=True), how="left")
        report["coverage"]["qualifying_match_rate"] = float(joined["has_quali"].notna().mean())
    if not races.empty and not results.empty:
        completed = results[["season", "round"]].drop_duplicates()
        race_join = races[["season", "round", "race_date"]].merge(
            completed.assign(has_results=True), how="left"
        )
        report["coverage"]["scheduled_races_with_results"] = int(
            race_join["has_results"].notna().sum()
        )
        report["coverage"]["scheduled_races_without_results"] = int(
            race_join["has_results"].isna().sum()
        )

    for name in ("laps", "weather", "session_results"):
        files = sorted(
            (paths.processed_fastf1 / name).glob("season=*/round=*/session=*/data.parquet")
        )
        report["fastf1"][name] = {
            "partitions": len(files),
            "rows": sum(len(pd.read_parquet(path)) for path in files),
        }

    report["prediction_cutoffs"] = {
        "pre_quali": "strictly before target qualifying start",
        "post_quali": "after target qualifying classification and known grid penalties",
        "same_race_outcomes": "labels only; prohibited from target-race features",
    }
    return report
