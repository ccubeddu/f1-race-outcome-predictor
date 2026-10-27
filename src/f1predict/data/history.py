"""Stage older seasons independently of production data and audit their usability."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.audit import build_audit
from f1predict.data.contracts import CONTRACTS, validate_table
from f1predict.data.io import atomic_json_write, atomic_text_write, read_partitioned_table
from f1predict.data.pipeline import sync_f1db, sync_jolpica


def _fingerprints(paths: DataPaths) -> dict[str, str]:
    files = [paths.catalog]
    for name in ("processed", "features", "models", "evaluation"):
        files.extend(sorted((paths.root / name).rglob("*")))
    return {
        str(path.relative_to(paths.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.is_file()
    }


def prepare_history_stage(production: DataPaths, stage: DataPaths) -> dict[str, Any]:
    """Copy a compact baseline once; subsequent runs resume the same snapshot."""
    origin, destination = production.root.resolve(), stage.root.resolve()
    if origin == destination or origin in destination.parents or destination in origin.parents:
        raise ValueError("Staging directory must be separate from the production data directory")
    manifest_path = destination / "baseline_snapshot.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["production_root"] != str(origin):
            raise ValueError("Stage belongs to a different production dataset")
        if manifest["production_fingerprints"] != _fingerprints(production):
            raise ValueError(
                "Production data changed after this stage was created; choose a new stage"
            )
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Staging directory is nonempty without a baseline snapshot; choose a new stage"
        )
    before = _fingerprints(production)
    results = read_partitioned_table(production.processed_jolpica, "results")
    if results.empty:
        raise ValueError("An existing production baseline is required before historical backfill")
    stage.create()
    for name in ("jolpica", "f1db", "openf1", "fastf1"):
        source = origin / "processed" / name
        if source.exists():
            shutil.copytree(source, destination / "processed" / name, dirs_exist_ok=True)
    if production.catalog.exists():
        shutil.copyfile(production.catalog, stage.catalog)
    archive = production.raw_f1db / "f1db-csv.zip"
    if archive.exists():
        shutil.copyfile(archive, stage.raw_f1db / archive.name)
        # Preserve useful source provenance without copying expiring signed URLs.
        source_meta = archive.with_suffix(".meta.json")
        if source_meta.exists():
            meta = json.loads(source_meta.read_text())
            atomic_json_write(
                stage.raw_f1db / source_meta.name,
                {
                    key: meta[key]
                    for key in ("sha256", "source_url", "license", "downloaded_at")
                    if key in meta
                },
            )
    if before != _fingerprints(production):
        raise RuntimeError("Production data changed while creating the baseline snapshot")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "production_root": str(origin),
        "production_fingerprints": before,
        "baseline_season_start": int(results["season"].min()),
        "baseline_season_end": int(results["season"].max()),
        "baseline_races": int(results[["season", "round"]].drop_duplicates().shape[0]),
        "baseline_driver_rows": len(results),
        "models_copied": False,
        "features_copied": False,
    }
    atomic_json_write(manifest_path, manifest)
    return manifest


def _name_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def audit_history_expansion(
    production: DataPaths, stage: DataPaths, *, start: int, end: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    snapshot = json.loads((stage.root / "baseline_snapshot.json").read_text())
    tables = {
        name: read_partitioned_table(stage.processed_jolpica, name)
        for name in ("races", "results", "qualifying", "sprints", "pit_stops")
    }
    for name, frame in tables.items():
        if frame.empty and "season" not in frame:
            tables[name] = frame.reindex(columns=CONTRACTS[name].required)
            frame = tables[name]
        validate_table(name, frame)
    races, results, qualifying = (tables[name] for name in ("races", "results", "qualifying"))
    practice = read_partitioned_table(stage.processed_f1db, "practice_results")
    validate_table("f1db_practice_results", practice)
    archive_path = stage.raw_f1db / "f1db-csv.zip"
    with zipfile.ZipFile(archive_path) as archive:
        independent_results = pd.read_csv(
            archive.open("f1db-races-race-results.csv"), low_memory=False
        )
        drivers = (
            pd.read_csv(archive.open("f1db-drivers.csv"))
            if "f1db-drivers.csv" in archive.namelist()
            else pd.DataFrame()
        )
    expected = independent_results.loc[
        independent_results["year"].between(start, end), ["year", "round"]
    ].drop_duplicates()
    actual = results.loc[results["season"].between(start, end), ["season", "round"]]
    actual = actual.drop_duplicates()
    expected_keys = set(expected.itertuples(index=False, name=None))
    actual_keys = set(actual.itertuples(index=False, name=None))
    expected_counts = independent_results.loc[independent_results["year"].between(start, end)]
    expected_counts = expected_counts.groupby(["year", "round"]).size().rename("f1db_rows")
    actual_counts = results.loc[results["season"].between(start, end)]
    actual_counts = actual_counts.groupby(["season", "round"]).size().rename("jolpica_rows")
    expected_counts.index.names = ["season", "round"]
    count_check = pd.concat([expected_counts, actual_counts], axis=1).reset_index()
    count_differences = count_check.loc[count_check["f1db_rows"].ne(count_check["jolpica_rows"])]
    excluded_entries = []
    if not drivers.empty:
        lookup = drivers.set_index("id")
        for mismatch in count_differences.itertuples():
            source_rows = independent_results.loc[
                independent_results["year"].eq(mismatch.season)
                & independent_results["round"].eq(mismatch.round)
            ].copy()
            source_rows["driver_code"] = source_rows["driverId"].map(lookup["abbreviation"])
            source_rows["driver_name"] = source_rows["driverId"].map(lookup["fullName"])
            actual_codes = results.loc[
                results["season"].eq(mismatch.season) & results["round"].eq(mismatch.round),
                "driver_code",
            ].dropna()
            absent = source_rows.loc[~source_rows["driver_code"].isin(actual_codes)]
            for entry in absent.to_dict("records"):
                excluded_entries.append(
                    {
                        "season": int(mismatch.season),
                        "round": int(mismatch.round),
                        "driver_id": entry["driverId"],
                        "driver_name": entry["driver_name"],
                        "source_position_text": entry["positionText"],
                        "non_race_entry": entry["positionText"] in {"DNQ", "DNP"},
                    }
                )

    rows = []
    for year in sorted(results["season"].unique()):
        result = results.loc[results["season"].eq(year)]
        quali = qualifying.loc[qualifying["season"].eq(year)]
        prac = practice.loc[practice["season"].eq(year)]
        key = ["season", "round", "driver_id"]
        matched = result[key].merge(quali[key].assign(has_quali=True), how="left", on=key)
        rows.append(
            {
                "season": int(year),
                "races": result["round"].nunique(),
                "driver_rows": len(result),
                "qualifying_rows": len(quali),
                "qualifying_match_rate": float(matched["has_quali"].notna().mean()),
                "practice_rows": len(prac),
                "practice_races": prac["round"].nunique(),
                "practice_timed_lap_rate": float(prac["best_lap_seconds"].notna().mean()),
                "pit_stop_rows": int(tables["pit_stops"]["season"].eq(year).sum()),
                "grid_position_rate": float(result["grid_position"].notna().mean()),
                "result_date_rate": float(result["race_date"].notna().mean()),
            }
        )
    coverage = pd.DataFrame(rows)
    old_practice = practice.loc[practice["season"].between(start, end)]
    old_results = results.loc[results["season"].between(start, end)].copy()
    car_number = pd.to_numeric(old_results["car_number"], errors="coerce").astype("Int64")
    permanent_number = pd.to_numeric(old_results["driver_number"], errors="coerce").astype("Int64")
    old_results["driver_number"] = car_number.fillna(permanent_number)
    old_results["race_driver_name"] = (
        old_results["given_name"].fillna("") + " " + old_results["family_name"].fillna("")
    )
    old_results["race_driver_code"] = old_results.get("driver_code", pd.Series(dtype="string"))
    identity = old_practice.merge(
        old_results[["season", "round", "driver_number", "race_driver_name", "race_driver_code"]],
        on=["season", "round", "driver_number"],
        how="left",
        validate="many_to_one",
    )
    matched_identity = identity["race_driver_name"].notna()
    practice_code = identity.get("driver_code", pd.Series(index=identity.index, dtype="string"))
    same_code = (
        practice_code.notna()
        & practice_code.ne("")
        & practice_code.eq(identity["race_driver_code"])
    ).fillna(False)
    substitutions = identity.loc[
        matched_identity
        & ~same_code
        & _name_key(identity["driver_name"]).ne(_name_key(identity["race_driver_name"]))
    ]
    sessions = old_practice[["season", "round", "session"]].drop_duplicates()
    sessions = sessions.merge(races, on=["season", "round"], validate="many_to_one")
    session_audit = []
    for record in sessions.to_dict("records"):
        prefix = record["session"].lower()
        day = pd.to_datetime(record.get(f"{prefix}_date"), errors="coerce")
        qday = pd.to_datetime(record.get("qualifying_date"), errors="coerce")
        exact = bool(
            pd.notna(record.get(f"{prefix}_time")) and pd.notna(record.get("qualifying_time"))
        )
        if pd.isna(day) or pd.isna(qday):
            status = "missing_schedule_dates"
        elif day < qday:
            status = "before_qualifying_calendar_day"
        elif day > qday:
            status = "after_qualifying_calendar_day"
        elif exact:
            status = "same_day_with_scheduled_times_not_publication_times"
        else:
            status = "same_day_without_exact_times"
        session_audit.append(
            {
                "season": record["season"],
                "round": record["round"],
                "session": record["session"],
                "timing_status": status,
            }
        )
    production_unchanged = snapshot["production_fingerprints"] == _fingerprints(production)
    total_races = len(results[["season", "round"]].drop_duplicates())
    errors = []
    if expected_keys != actual_keys:
        errors.append("Backfilled race keys differ from the independent F1DB result inventory")
    if not production_unchanged:
        errors.append("Production data or artifacts changed during staging")
    if set(range(start, end + 1)) - set(actual["season"]):
        errors.append("Requested seasons are missing")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": str(stage.root.resolve()),
        "status": "passed_with_coverage_caveats" if not errors else "failed",
        "errors": errors,
        "requested_years": [start, end],
        "baseline_races": snapshot["baseline_races"],
        "candidate_races": total_races,
        "added_races": total_races - snapshot["baseline_races"],
        "baseline_driver_rows": snapshot["baseline_driver_rows"],
        "candidate_driver_rows": len(results),
        "added_driver_rows": len(old_results),
        "production_unchanged": production_unchanged,
        "independent_expected_backfilled_races": len(expected_keys),
        "missing_race_keys": sorted(expected_keys - actual_keys),
        "unexpected_race_keys": sorted(actual_keys - expected_keys),
        "independent_driver_count_differences": count_differences.to_dict("records"),
        "independent_missing_entry_details": excluded_entries,
        "practice": {
            "backfilled_rows": len(old_practice),
            "timed_lap_rate": float(old_practice["best_lap_seconds"].notna().mean()),
            "source_session_date_rate": float(old_practice["session_date"].notna().mean()),
            "source_session_time_rate": float(old_practice["session_time"].notna().mean()),
            "timing_status_counts": pd.DataFrame(session_audit)["timing_status"]
            .value_counts()
            .to_dict(),
            "session_timing_audit": session_audit,
            "same_number_different_driver_rows": len(substitutions),
            "substitution_examples": substitutions[
                ["season", "round", "session", "driver_name", "race_driver_name", "driver_number"]
            ]
            .head(15)
            .to_dict("records"),
            "practice_only_rows": int((~matched_identity).sum()),
        },
        "source_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "source_urls": {
            "jolpica": "https://api.jolpi.ca/ergast/f1/",
            "f1db": "https://github.com/f1db/f1db",
        },
        "data_audit": build_audit(stage),
        "training_performed": False,
        "next_stage": "Resolve practice timing and driver substitutions before feature promotion",
    }, coverage


def backfill_history(
    production: DataPaths,
    stage: DataPaths,
    output: Path,
    *,
    start: int = 2010,
    end: int = 2017,
    progress: Callable[[int, dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    baseline = prepare_history_stage(production, stage)
    if start < 1950 or end < start or end >= baseline["baseline_season_start"]:
        raise ValueError("Backfill range must precede the first production season")
    sync_jolpica(stage, start, end, progress=progress)
    sync_f1db(stage, start, end)
    report, coverage = audit_history_expansion(production, stage, start=start, end=end)
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output / "season_coverage.csv", index=False)
    atomic_json_write(output / "audit.json", report)
    table = (
        "| Season | Races | Driver rows | Qualifying match | "
        "Practice rows | Timed laps | Pit-stop rows |\n"
    )
    table += "|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in coverage.itertuples():
        table += (
            f"| {row.season} | {row.races} | {row.driver_rows} | "
            f"{row.qualifying_match_rate:.1%} | {row.practice_rows} | "
            f"{row.practice_timed_lap_rate:.1%} | {row.pit_stop_rows} |\n"
        )
    non_race_count = sum(
        entry["non_race_entry"] for entry in report["independent_missing_entry_details"]
    )
    atomic_text_write(
        output / "report.md",
        (
            "# Historical data expansion — stage 1\n\n"
            f"Status: {report['status']}. "
            f"Production unchanged: {production_unchanged_text(report)}.\n\n"
            f"Completed races: **{report['baseline_races']} → {report['candidate_races']}** "
            f"(+{report['added_races']}). Driver-race rows: "
            f"**{report['baseline_driver_rows']} → {report['candidate_driver_rows']}**.\n\n"
            "This is a staged data expansion. Model training, features, predictions and evaluation "
            "artifacts in the production data directory were not replaced.\n\n"
            "## Season coverage\n\n" + table + "\n\n"
            "## Coverage caveats before feature promotion\n\n"
            f"Older practice classifications: {report['practice']['backfilled_rows']:,} rows; "
            f"timed laps {report['practice']['timed_lap_rate']:.1%}. "
            f"Practice-source date coverage {report['practice']['source_session_date_rate']:.1%}; "
            f"time coverage {report['practice']['source_session_time_rate']:.1%}.\n\n"
            "Timing evidence from the independent Jolpica schedule: "
            f"{json.dumps(report['practice']['timing_status_counts'], sort_keys=True)}. "
            "Scheduled times do not prove actual completion/publication. Missing dates and "
            "same-day sessions without times need a conservative feature policy.\n\n"
            f"Detected {report['practice']['same_number_different_driver_rows']} practice rows "
            "where the same car number belongs to a different driver in the race results. "
            "A car-number-only join can attribute a substitute's practice to the race driver; "
            "this must be fixed in the next feature stage.\n\n"
            "Jolpica pit-stop data starts in 2011; 2010 is recorded as unavailable, not zero pit "
            "stops. Detailed tyre/lap telemetry does not become available simply by extending "
            "the classification history.\n\n"
            f"Driver-entry counts differ between the sources in "
            f"{len(report['independent_driver_count_differences'])} races. "
            "Per-race differences are retained in audit.json for reconciliation.\n\n"
            f"Source-only DNQ/DNP entries identified: {non_race_count}. "
            "These are kept as source-audit evidence rather than assigned "
            "invented finishing positions.\n\n"
            "Sources: [Jolpica](https://github.com/jolpica/jolpica-f1), "
            "[F1DB, CC-BY-4.0](https://github.com/f1db/f1db). "
            f"F1DB archive SHA-256: `{report['source_sha256']}`.\n\n"
            "## Next stage\n\n"
            "Audit conservative practice eligibility, identity joins, normalized "
            "historical features "
            "and recency weighting in the staging dataset. Compare 2018+, 2014+ and 2010+ using "
            "identical chronological evaluation races and production model settings. No accuracy "
            "gain is claimed at this data-only stage.\n"
        ),
    )
    if report["errors"]:
        raise RuntimeError("Backfill audit failed; inspect the saved audit.json")
    return report


def production_unchanged_text(report: dict[str, Any]) -> str:
    return "yes (SHA-256 verified)" if report["production_unchanged"] else "NO"
