from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.data.io import read_parquet_files, read_partitioned_table

PRACTICE_RECIPE = "identity_completed_session_v2"
DRIVER_KEYS = ["season", "round", "driver_id"]
SESSION_KEYS = ["season", "round", "session"]


def _scheduled_pre_quali_sessions(
    schedule: pd.DataFrame,
    sessions: pd.DataFrame | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Require a source end before Q, or a strictly earlier calendar day.

    A start time cannot prove completion. Without end timestamps we deliberately
    discard same-day practice. Archives establish event time, not publication time.
    Both prediction modes share this conservative practice cutoff.
    """
    rows: list[pd.DataFrame] = []
    for session in ("FP1", "FP2", "FP3"):
        label = session.lower()
        selected = schedule[
            [
                "season",
                "round",
                f"{label}_date",
                "qualifying_date",
                "qualifying_time",
            ]
        ].copy()
        selected["session"] = session
        selected["qualifying_scheduled_at"] = pd.to_datetime(
            selected["qualifying_date"].fillna("").astype(str)
            + " "
            + selected["qualifying_time"].fillna("").astype(str),
            errors="coerce",
            format="mixed",
            utc=True,
        )
        now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        now = now.tz_localize("UTC") if now.tzinfo is None else now
        selected["cutoff"] = selected["qualifying_scheduled_at"].clip(upper=now)
        day = pd.to_datetime(selected[f"{label}_date"], errors="coerce", utc=True)
        selected["is_pre_quali"] = day.lt(selected["cutoff"].dt.normalize())
        selected["timing_evidence"] = np.where(
            selected["is_pre_quali"], "earlier_calendar_day", "unverified_or_after_cutoff"
        )
        rows.append(selected[[*SESSION_KEYS, "cutoff", "is_pre_quali", "timing_evidence"]])
    eligibility = pd.concat(rows, ignore_index=True)
    if sessions is not None and not sessions.empty:
        ends = sessions[[*SESSION_KEYS, "date_end"]].copy()
        if ends.duplicated(SESSION_KEYS).any():
            raise ValueError("Ambiguous practice sessions for one race/session")
        ends["verified_end"] = pd.to_datetime(ends.pop("date_end"), errors="coerce", utc=True)
        eligibility = eligibility.merge(ends, on=SESSION_KEYS, how="left", validate="one_to_one")
        known = eligibility["verified_end"].notna()
        eligibility.loc[known, "is_pre_quali"] = eligibility.loc[known, "verified_end"].lt(
            eligibility.loc[known, "cutoff"]
        )
        eligibility.loc[known, "timing_evidence"] = "source_session_end"
    return eligibility


def _name_key(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def match_practice_drivers(frame: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Match unique, agreeing identity evidence against event entrants, NEVER car number.

    Unmatched substitutes stay unmatched. Only entrant identifiers are consulted;
    no target outcomes are used to resolve identities.
    """
    result = frame.drop(columns="driver_id", errors="ignore").copy()
    result["driver_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if frame.empty:
        return result
    votes = []
    for source, target in (
        ("driver_code", "driver_code"),
        ("name_acronym", "driver_code"),
        ("driver_name", "driver_name"),
        ("full_name", "driver_name"),
    ):
        if source not in frame or target not in roster:
            continue
        lookup = roster[["season", "round", "driver_id", target]].copy()
        lookup["_identity"] = _name_key(lookup.pop(target))
        lookup = lookup.loc[lookup["_identity"].ne("")].drop_duplicates()
        lookup = lookup.loc[~lookup.duplicated(["season", "round", "_identity"], keep=False)]
        query = frame[["season", "round"]].copy()
        query["_identity"] = _name_key(frame[source])
        matched = query.merge(
            lookup, on=["season", "round", "_identity"], how="left", validate="many_to_one"
        )["driver_id"]
        matched.index = frame.index
        votes.append(matched)
    if votes:
        evidence = pd.concat(votes, axis=1).astype("string")
        unique = evidence.nunique(axis=1).eq(1)
        result.loc[unique, "driver_id"] = evidence.bfill(axis=1).iloc[:, 0].loc[unique]
    return result


def _eligible(frame: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.merge(
        eligibility[[*SESSION_KEYS, "is_pre_quali", "cutoff"]],
        on=SESSION_KEYS,
        how="left",
        validate="many_to_one",
    )
    allowed = result.pop("is_pre_quali").fillna(False).astype(bool)
    for timestamp in ("date", "date_end"):
        if timestamp in result:
            at = pd.to_datetime(result[timestamp], errors="coerce", format="mixed", utc=True)
            allowed &= at.isna() | at.lt(result["cutoff"])
    if "date_start" in result and "lap_duration" in result:
        end = pd.to_datetime(result["date_start"], errors="coerce", format="mixed", utc=True)
        end += pd.to_timedelta(result["lap_duration"], unit="s")
        allowed &= end.isna() | end.lt(result["cutoff"])
    return result.loc[allowed].drop(columns="cutoff")


def _wide(
    frame: pd.DataFrame,
    index: list[str],
    values: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=index)
    if frame.duplicated([*index, "session"]).any():
        raise ValueError("Duplicate practice identities; refusing to choose an arbitrary row")
    pivot = frame.pivot(
        index=index,
        columns="session",
        values=values,
    )
    pivot.columns = [f"{prefix}_{session.lower()}_{value}" for value, session in pivot.columns]
    return pivot.reset_index()


def f1db_practice_features(
    processed_f1db: Path,
    eligibility: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    frame = _eligible(read_partitioned_table(processed_f1db, "practice_results"), eligibility)
    if frame.empty:
        return pd.DataFrame(columns=DRIVER_KEYS)
    group = ["season", "round", "session"]
    session_best = frame.groupby(group, observed=True)["best_lap_seconds"].transform("min")
    field_size = frame.groupby(group, observed=True)["driver_id"].transform("size")
    frame["gap_pct"] = frame["best_lap_seconds"].div(session_best).sub(1)
    frame["position_pct"] = frame["position"].sub(1).div(field_size.sub(1).clip(lower=1))
    frame["has_timed_lap"] = frame["best_lap_seconds"].notna().astype("int8")
    frame = match_practice_drivers(frame, roster).dropna(subset="driver_id")
    return _wide(
        frame,
        DRIVER_KEYS,
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


def _long_run_features(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Tyre-conditioned long-run proxy, NOT a fuel-corrected race-pace estimate.

    Require >=5 clean, non-boundary laps in one stint. Compare compound/session/
    five-lap tyre-age bands with >=3 drivers. Missing ages and sparse peers stay missing.
    """
    keys = [*SESSION_KEYS, "session_key", "driver_number"]
    required = {"lap_start", "lap_end", "stint_number", "compound", "tyre_age_at_start"}
    if laps.empty or stints.empty or not required.issubset(stints):
        return pd.DataFrame(columns=keys)
    joined = laps.merge(
        stints[[*keys, *sorted(required)]], on=keys, how="inner", validate="many_to_many"
    )
    joined = joined.loc[
        joined["lap_number"].gt(joined["lap_start"]) & joined["lap_number"].lt(joined["lap_end"])
    ].copy()
    joined = joined.loc[~joined.duplicated([*keys, "lap_number"], keep=False)]
    joined["tyre_age"] = joined["tyre_age_at_start"] + joined["lap_number"] - joined["lap_start"]
    joined = joined.loc[
        joined["tyre_age"].ge(0) & joined["compound"].isin(["SOFT", "MEDIUM", "HARD"])
    ]
    # Qualifying-style short runs must not establish the long-run reference pace.
    run_size = joined.groupby([*keys, "stint_number"], observed=True)["lap_number"].transform(
        "size"
    )
    joined = joined.loc[run_size.ge(5)].copy()
    joined["_age_band"] = joined["tyre_age"].floordiv(5)
    peers = ["session_key", "compound", "_age_band"]
    peer = joined.groupby([*peers, "driver_number"], observed=True)["lap_duration"].median()
    peer = peer.reset_index()
    peer["_reference"] = peer.groupby(peers, observed=True)["lap_duration"].transform("median")
    peer["_peers"] = peer.groupby(peers, observed=True)["driver_number"].transform("nunique")
    joined = joined.merge(
        peer[[*peers, "driver_number", "_reference", "_peers"]],
        on=[*peers, "driver_number"],
        validate="many_to_one",
    )
    joined["_gap"] = joined["lap_duration"].div(joined["_reference"]).sub(1)
    joined.loc[joined["_peers"].lt(3), "_gap"] = np.nan
    runs = (
        joined.groupby([*keys, "stint_number"], observed=True)
        .agg(
            detail_long_run_clean_laps=("lap_duration", "size"),
            detail_tyre_adjusted_gap_pct=("_gap", "median"),
            detail_long_run_tyre_age=("tyre_age", "median"),
        )
        .reset_index()
    )
    runs = runs.loc[runs["detail_long_run_clean_laps"].ge(5)]
    runs = runs.sort_values(["detail_long_run_clean_laps", "stint_number"], ascending=[False, True])
    return runs.drop_duplicates(keys).drop(columns="stint_number")


def openf1_practice_features(
    processed_openf1: Path,
    eligibility: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    laps = _eligible(_openf1_table(processed_openf1, "practice_laps"), eligibility)
    if laps.empty:
        return pd.DataFrame(columns=DRIVER_KEYS)
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
    identity_cols = [c for c in ("name_acronym", "full_name") if c in laps]
    identities = laps[[*keys, *identity_cols]].drop_duplicates()
    if identities.duplicated(keys).any():
        raise ValueError("Conflicting driver identities within an OpenF1 session")
    detail = detail.merge(identities, on=keys, validate="one_to_one")
    session_best = detail.groupby("session_key", observed=True)["detail_best_lap"].transform("min")
    session_median = detail.groupby("session_key", observed=True)["detail_median_lap"].transform(
        "min"
    )
    detail["detail_best_gap_pct"] = detail["detail_best_lap"].div(session_best).sub(1)
    # All-lap median is not a true long run; retain it under an honest feature name.
    detail["detail_all_lap_median_gap_pct"] = detail["detail_median_lap"].div(session_median).sub(1)
    detail["detail_consistency_cv"] = detail["detail_lap_std"].div(detail["detail_median_lap"])

    stints = _eligible(_openf1_table(processed_openf1, "practice_stints"), eligibility)
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
        detail = detail.merge(
            _long_run_features(laps, stints), on=keys, how="left", validate="one_to_one"
        )

    detail = match_practice_drivers(detail, roster).dropna(subset="driver_id")
    value_columns = [column for column in detail if column.startswith("detail_")]
    wide = _wide(
        detail,
        DRIVER_KEYS,
        value_columns,
        prefix="openf1",
    )

    weather = _eligible(_openf1_table(processed_openf1, "practice_weather"), eligibility)
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
    entrants = (
        roster[DRIVER_KEYS]
        .drop_duplicates()
        .merge(weather_wide, on=["season", "round"], how="inner", validate="many_to_one")
    )
    return wide.merge(entrants, on=DRIVER_KEYS, how="outer", validate="one_to_one")


def add_practice_features(
    base: pd.DataFrame,
    processed_f1db: Path,
    processed_openf1: Path,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    frame = base.copy()
    sessions = read_partitioned_table(processed_openf1, "practice_sessions")
    eligibility = _scheduled_pre_quali_sessions(schedule, sessions)
    f1db = f1db_practice_features(processed_f1db, eligibility, base)
    if not f1db.empty:
        frame = frame.merge(
            f1db,
            on=DRIVER_KEYS,
            how="left",
            validate="one_to_one",
        )
    openf1 = openf1_practice_features(processed_openf1, eligibility, base)
    if not openf1.empty:
        frame = frame.merge(
            openf1,
            on=DRIVER_KEYS,
            how="left",
            validate="one_to_one",
        )
    practice_columns = [
        column for column in frame if column.startswith("practice_") or column.startswith("openf1_")
    ]
    frame["practice_data_available"] = frame[practice_columns].notna().any(axis=1).astype("int8")
    frame["openf1_detail_available"] = (
        frame[[column for column in practice_columns if "_detail_" in column]]
        .notna()
        .any(axis=1)
        .astype("int8")
        if any(column.startswith("openf1_") for column in practice_columns)
        else np.int8(0)
    )
    long_run = [c for c in practice_columns if c.endswith("tyre_adjusted_gap_pct")]
    frame["practice_long_run_available"] = frame[long_run].notna().any(axis=1).astype("int8")
    return frame
