from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.data.io import read_parquet_files, read_partitioned_table


def _scheduled_pre_quali_sessions(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return only practice sessions scheduled no later than qualifying day/time."""
    rows: list[pd.DataFrame] = []
    for session in ("FP1", "FP2", "FP3"):
        label = session.lower()
        selected = schedule[
            [
                "season",
                "round",
                f"{label}_date",
                f"{label}_time",
                "qualifying_date",
                "qualifying_time",
            ]
        ].copy()
        selected["session"] = session
        selected["session_scheduled_at"] = pd.to_datetime(
            selected[f"{label}_date"].fillna("").astype(str)
            + " "
            + selected[f"{label}_time"].fillna("").astype(str),
            errors="coerce",
            format="mixed",
            utc=True,
        )
        selected["qualifying_scheduled_at"] = pd.to_datetime(
            selected["qualifying_date"].fillna("").astype(str)
            + " "
            + selected["qualifying_time"].fillna("").astype(str),
            errors="coerce",
            format="mixed",
            utc=True,
        )
        selected["session_date_only"] = pd.to_datetime(selected[f"{label}_date"], errors="coerce")
        selected["qualifying_date_only"] = pd.to_datetime(
            selected["qualifying_date"], errors="coerce"
        )
        exact = (
            selected[f"{label}_time"].fillna("").astype(str).str.len().gt(0)
            & selected["qualifying_time"].fillna("").astype(str).str.len().gt(0)
            & selected["session_scheduled_at"].notna()
            & selected["qualifying_scheduled_at"].notna()
        )
        selected["is_pre_quali"] = (
            exact & selected["session_scheduled_at"].lt(selected["qualifying_scheduled_at"])
        ) | (
            ~exact
            & selected["session_date_only"].notna()
            & selected["qualifying_date_only"].notna()
            & selected["session_date_only"].le(selected["qualifying_date_only"])
        )
        rows.append(selected[["season", "round", "session", "is_pre_quali"]])
    return pd.concat(rows, ignore_index=True)


def _wide(
    frame: pd.DataFrame,
    index: list[str],
    values: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=index)
    pivot = frame.pivot_table(
        index=index,
        columns="session",
        values=values,
        aggfunc="first",
        observed=True,
    )
    pivot.columns = [f"{prefix}_{session.lower()}_{value}" for value, session in pivot.columns]
    return pivot.reset_index()


def f1db_practice_features(processed_f1db: Path, schedule: pd.DataFrame) -> pd.DataFrame:
    frame = read_partitioned_table(processed_f1db, "practice_results")
    if frame.empty:
        return pd.DataFrame(columns=["season", "round", "driver_number"])
    eligibility = _scheduled_pre_quali_sessions(schedule)
    frame = frame.merge(
        eligibility,
        on=["season", "round", "session"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.loc[frame.pop("is_pre_quali").fillna(False)].copy()
    group = ["season", "round", "session"]
    session_best = frame.groupby(group, observed=True)["best_lap_seconds"].transform("min")
    field_size = frame.groupby(group, observed=True)["driver_id"].transform("size")
    frame["gap_pct"] = frame["best_lap_seconds"].div(session_best).sub(1)
    frame["position_pct"] = frame["position"].sub(1).div(field_size.sub(1).clip(lower=1))
    frame["has_timed_lap"] = frame["best_lap_seconds"].notna().astype("int8")
    return _wide(
        frame,
        ["season", "round", "driver_number"],
        [
            "position",
            "position_pct",
            "gap_pct",
            "laps_completed",
            "has_timed_lap",
        ],
        prefix="practice",
    )


def _openf1_table(root: Path, name: str) -> pd.DataFrame:
    files = sorted((root / name).glob("season=*/round=*/session=*/data.parquet"))
    return read_parquet_files(files)


def openf1_practice_features(processed_openf1: Path, schedule: pd.DataFrame) -> pd.DataFrame:
    laps = _openf1_table(processed_openf1, "practice_laps")
    if laps.empty:
        return pd.DataFrame(columns=["season", "round", "driver_number"])
    eligibility = _scheduled_pre_quali_sessions(schedule)
    laps = laps.merge(
        eligibility,
        on=["season", "round", "session"],
        how="left",
        validate="many_to_one",
    )
    laps = laps.loc[laps.pop("is_pre_quali").fillna(False)].copy()
    laps = laps.loc[laps["lap_duration"].notna() & laps["lap_duration"].gt(0)].copy()
    if "is_pit_out_lap" in laps:
        laps = laps.loc[~laps["is_pit_out_lap"].fillna(False).astype(bool)]
    driver_best = laps.groupby(["session_key", "driver_number"], observed=True)[
        "lap_duration"
    ].transform("min")
    laps = laps.loc[laps["lap_duration"].le(driver_best.mul(1.25))]
    keys = ["season", "round", "session", "session_key", "driver_number"]
    aggregations: dict[str, tuple[str, str]] = {
        "detail_best_lap": ("lap_duration", "min"),
        "detail_median_lap": ("lap_duration", "median"),
        "detail_lap_std": ("lap_duration", "std"),
        "detail_clean_laps": ("lap_duration", "size"),
    }
    for source, target in (
        ("duration_sector_1", "detail_best_sector1"),
        ("duration_sector_2", "detail_best_sector2"),
        ("duration_sector_3", "detail_best_sector3"),
        ("st_speed", "detail_max_speed"),
    ):
        if source in laps:
            aggregations[target] = (source, "max" if source == "st_speed" else "min")
    detail = laps.groupby(keys, observed=True).agg(**aggregations).reset_index()
    session_best = detail.groupby("session_key", observed=True)["detail_best_lap"].transform("min")
    session_median = detail.groupby("session_key", observed=True)["detail_median_lap"].transform(
        "min"
    )
    detail["detail_best_gap_pct"] = detail["detail_best_lap"].div(session_best).sub(1)
    detail["detail_long_run_gap_pct"] = detail["detail_median_lap"].div(session_median).sub(1)
    detail["detail_consistency_cv"] = detail["detail_lap_std"].div(detail["detail_median_lap"])

    stints = _openf1_table(processed_openf1, "practice_stints")
    if not stints.empty:
        stints["stint_laps"] = stints["lap_end"].sub(stints["lap_start"]).add(1)
        stint_summary = (
            stints.groupby(keys, observed=True)
            .agg(
                detail_stints=("stint_number", "nunique"),
                detail_max_stint_laps=("stint_laps", "max"),
                detail_compounds=("compound", "nunique"),
            )
            .reset_index()
        )
        detail = detail.merge(stint_summary, on=keys, how="left", validate="one_to_one")

    value_columns = [column for column in detail if column.startswith("detail_")]
    wide = _wide(
        detail,
        ["season", "round", "driver_number"],
        value_columns,
        prefix="openf1",
    )

    weather = _openf1_table(processed_openf1, "practice_weather")
    if weather.empty:
        return wide
    weather_aggregations: dict[str, tuple[str, str]] = {}
    for source, target, function in (
        ("air_temperature", "weather_air_temp", "mean"),
        ("track_temperature", "weather_track_temp", "mean"),
        ("humidity", "weather_humidity", "mean"),
        ("rainfall", "weather_rain_fraction", "mean"),
        ("wind_speed", "weather_wind_speed", "mean"),
    ):
        if source in weather:
            weather_aggregations[target] = (source, function)
    weather_keys = ["season", "round", "session", "session_key"]
    weather_summary = (
        weather.groupby(weather_keys, observed=True).agg(**weather_aggregations).reset_index()
    )
    weather_wide = _wide(
        weather_summary,
        ["season", "round"],
        list(weather_aggregations),
        prefix="openf1",
    )
    return wide.merge(weather_wide, on=["season", "round"], how="left")


def add_practice_features(
    base: pd.DataFrame,
    processed_f1db: Path,
    processed_openf1: Path,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    frame = base.copy()
    f1db = f1db_practice_features(processed_f1db, schedule)
    if not f1db.empty:
        frame = frame.merge(
            f1db,
            on=["season", "round", "driver_number"],
            how="left",
            validate="one_to_one",
        )
    openf1 = openf1_practice_features(processed_openf1, schedule)
    if not openf1.empty:
        frame = frame.merge(
            openf1,
            on=["season", "round", "driver_number"],
            how="left",
            validate="one_to_one",
        )
    practice_columns = [
        column for column in frame if column.startswith("practice_") or column.startswith("openf1_")
    ]
    frame["practice_data_available"] = frame[practice_columns].notna().any(axis=1).astype("int8")
    frame["openf1_detail_available"] = (
        frame[[column for column in practice_columns if column.startswith("openf1_")]]
        .notna()
        .any(axis=1)
        .astype("int8")
        if any(column.startswith("openf1_") for column in practice_columns)
        else np.int8(0)
    )
    return frame
