from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write, read_partitioned_table
from f1predict.features.practice import add_practice_features
from f1predict.features.rolling import (
    prior_count,
    shifted_expanding_mean,
    shifted_rolling_mean,
    shifted_rolling_std,
)

IDENTIFIERS = [
    "season",
    "round",
    "race_date",
    "race_name",
    "circuit_id",
    "driver_id",
    "driver_code",
    "driver_name",
    "driver_number",
    "constructor_id",
    "constructor_name",
]

TARGETS = [
    "target_finish_position",
    "target_win",
    "target_podium",
    "target_points",
    "target_dnf",
    "target_dns",
    "target_dsq",
    "target_classified",
]

FORBIDDEN_PRE_QUALI_PREFIXES = ("post_", "target_")


def _classified_status(status: pd.Series) -> pd.Series:
    text = status.fillna("").astype(str)
    return text.eq("Finished") | text.eq("Lapped") | text.str.match(r"^\+\d+ Laps?$")


def _base_table(paths: DataPaths) -> pd.DataFrame:
    results = read_partitioned_table(paths.processed_jolpica, "results")
    qualifying = read_partitioned_table(paths.processed_jolpica, "qualifying")
    if results.empty:
        raise RuntimeError("No race results found; run `f1predict data sync` first.")
    results["race_date"] = pd.to_datetime(results["race_date"], errors="coerce")
    results["driver_number"] = results["car_number"].combine_first(results["driver_number"])
    results["driver_name"] = (
        results["given_name"]
        .fillna("")
        .str.cat(results["family_name"].fillna(""), sep=" ")
        .str.strip()
    )
    qualifying_columns = [
        "season",
        "round",
        "driver_id",
        "quali_position",
        "q1_seconds",
        "q2_seconds",
        "q3_seconds",
    ]
    frame = results.merge(
        qualifying[qualifying_columns],
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    frame["field_size"] = frame.groupby(["season", "round"], observed=True)["driver_id"].transform(
        "size"
    )
    race_laps = frame.groupby(["season", "round"], observed=True)["laps_completed"].transform("max")
    frame["completion_fraction"] = frame["laps_completed"].div(race_laps.replace(0, np.nan))
    classified = _classified_status(frame["status"])
    dns = frame["status"].isin(["Did not start", "Withdrew"])
    dsq = frame["status"].eq("Disqualified")
    frame["target_finish_position"] = frame["finish_position"].astype("Int64")
    frame["target_win"] = frame["finish_position"].eq(1).astype("int8")
    frame["target_podium"] = frame["finish_position"].le(3).astype("int8")
    frame["target_points"] = frame["points"].gt(0).astype("int8")
    frame["target_classified"] = classified.astype("int8")
    frame["target_dns"] = dns.astype("int8")
    frame["target_dsq"] = dsq.astype("int8")
    frame["target_dnf"] = (~classified & ~dns & ~dsq).astype("int8")
    frame["season_round"] = pd.to_numeric(frame["round"], errors="coerce")
    frame["static_latitude"] = frame["latitude"]
    frame["static_longitude"] = frame["longitude"]
    frame["static_circuit_id"] = frame["circuit_id"].astype("string")
    frame["static_country"] = frame["country"].astype("string")
    date_of_birth = pd.to_datetime(frame["driver_dob"], errors="coerce")
    frame["static_driver_age"] = frame["race_date"].sub(date_of_birth).dt.days.div(365.2425)
    return frame.sort_values(["race_date", "season", "round", "driver_id"], ignore_index=True)


def _add_driver_history(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["driver_prior_starts"] = prior_count(data, "driver_id")
    for source, label in (
        ("target_finish_position", "finish"),
        ("points", "points"),
        ("target_dnf", "dnf"),
        ("quali_position", "quali"),
        ("grid_position", "grid"),
        ("completion_fraction", "completion"),
    ):
        for window in (3, 5, 10):
            data[f"driver_{label}_mean_{window}"] = shifted_rolling_mean(
                data, "driver_id", source, window
            )
        if source in {"target_finish_position", "quali_position"}:
            data[f"driver_{label}_std_5"] = shifted_rolling_std(data, "driver_id", source, 5)
    for source, target in (
        ("target_win", "driver_prior_wins"),
        ("target_podium", "driver_prior_podiums"),
        ("points", "driver_prior_points"),
        ("target_dnf", "driver_prior_dnfs"),
    ):
        data[target] = data.groupby("driver_id", sort=False, observed=True)[source].transform(
            lambda values: values.shift(1).fillna(0).cumsum()
        )
    return data


def _team_race_table(frame: pd.DataFrame) -> pd.DataFrame:
    team = (
        frame.groupby(
            ["season", "round", "race_date", "constructor_id"], observed=True, as_index=False
        )
        .agg(
            team_finish_mean=("target_finish_position", "mean"),
            team_points=("points", "sum"),
            team_dnf_rate=("target_dnf", "mean"),
            team_quali_mean=("quali_position", "mean"),
            team_completion_mean=("completion_fraction", "mean"),
        )
        .sort_values(["race_date", "constructor_id"], ignore_index=True)
    )
    team["constructor_prior_races"] = prior_count(team, "constructor_id")
    for source, label in (
        ("team_finish_mean", "finish"),
        ("team_points", "points"),
        ("team_dnf_rate", "dnf"),
        ("team_quali_mean", "quali"),
        ("team_completion_mean", "completion"),
    ):
        for window in (3, 5, 10):
            team[f"constructor_{label}_mean_{window}"] = shifted_rolling_mean(
                team, "constructor_id", source, window
            )
    team["constructor_season_points_pre"] = team.groupby(
        ["season", "constructor_id"], observed=True
    )["team_points"].transform(lambda values: values.shift(1).fillna(0).cumsum())
    feature_columns = [
        column
        for column in team
        if column.startswith("constructor_") and column != "constructor_id"
    ]
    return team[["season", "round", "constructor_id", *feature_columns]]


def _add_teammate_history(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    groups = data.groupby(["season", "round", "constructor_id"], observed=True)
    count = groups["driver_id"].transform("size")
    teammate_finish = (
        groups["target_finish_position"]
        .transform("sum")
        .sub(data["target_finish_position"])
        .div(count.sub(1).replace(0, np.nan))
    )
    teammate_quali = (
        groups["quali_position"]
        .transform("sum")
        .sub(data["quali_position"])
        .div(count.sub(1).replace(0, np.nan))
    )
    data["_teammate_finish_win"] = data["target_finish_position"].lt(teammate_finish).astype(float)
    data.loc[teammate_finish.isna(), "_teammate_finish_win"] = np.nan
    data["_teammate_quali_win"] = data["quali_position"].lt(teammate_quali).astype(float)
    data.loc[teammate_quali.isna(), "_teammate_quali_win"] = np.nan
    data["_teammate_finish_delta"] = data["target_finish_position"].sub(teammate_finish)
    data["_teammate_quali_delta"] = data["quali_position"].sub(teammate_quali)
    for source, target in (
        ("_teammate_finish_win", "teammate_finish_h2h_rate_10"),
        ("_teammate_quali_win", "teammate_quali_h2h_rate_10"),
        ("_teammate_finish_delta", "teammate_finish_delta_mean_10"),
        ("_teammate_quali_delta", "teammate_quali_delta_mean_10"),
    ):
        data[target] = shifted_rolling_mean(data, "driver_id", source, 10)
    return data.drop(
        columns=[
            "_teammate_finish_win",
            "_teammate_quali_win",
            "_teammate_finish_delta",
            "_teammate_quali_delta",
        ]
    )


def _add_track_history(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    groups = ["driver_id", "circuit_id"]
    data["track_prior_starts"] = prior_count(data, groups)
    for source, target in (
        ("target_finish_position", "track_finish_mean"),
        ("points", "track_points_mean"),
        ("target_dnf", "track_dnf_rate"),
        ("quali_position", "track_quali_mean"),
    ):
        data[target] = shifted_expanding_mean(data, groups, source)
    return data


def _add_championship_context(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["driver_season_points_pre"] = data.groupby(["season", "driver_id"], observed=True)[
        "points"
    ].transform(lambda values: values.shift(1).fillna(0).cumsum())
    data["driver_season_rank_pre"] = data.groupby(["season", "round"], observed=True)[
        "driver_season_points_pre"
    ].rank(method="min", ascending=False)
    return data


def _add_pit_history(frame: pd.DataFrame, paths: DataPaths) -> pd.DataFrame:
    pits = read_partitioned_table(paths.processed_jolpica, "pit_stops")
    if pits.empty:
        return frame
    pits = pits.loc[pits["duration_seconds"].between(12, 120)].copy()
    driver_pits = (
        pits.groupby(["season", "round", "driver_id"], observed=True)
        .agg(
            _pit_median=("duration_seconds", "median"),
            _pit_min=("duration_seconds", "min"),
            _pit_count=("stop", "count"),
        )
        .reset_index()
    )
    data = frame.merge(
        driver_pits,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    for source, target in (
        ("_pit_median", "driver_pit_median_mean_5"),
        ("_pit_min", "driver_pit_min_mean_5"),
        ("_pit_count", "driver_pit_count_mean_5"),
    ):
        data[target] = shifted_rolling_mean(data, "driver_id", source, 5)
    team_pits = (
        data.groupby(["season", "round", "race_date", "constructor_id"], observed=True)
        .agg(_team_pit_median=("_pit_median", "mean"))
        .reset_index()
        .sort_values(["race_date", "constructor_id"], ignore_index=True)
    )
    team_pits["constructor_pit_median_mean_5"] = shifted_rolling_mean(
        team_pits, "constructor_id", "_team_pit_median", 5
    )
    data = data.merge(
        team_pits[["season", "round", "constructor_id", "constructor_pit_median_mean_5"]],
        on=["season", "round", "constructor_id"],
        how="left",
        validate="many_to_one",
    )
    return data.drop(columns=["_pit_median", "_pit_min", "_pit_count"])


def _add_post_quali_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["post_quali_position"] = data["quali_position"]
    data["post_grid_position"] = data["grid_position"]
    data["post_started_pit_lane"] = data["grid_position"].eq(0).astype("int8")
    data["post_grid_position_effective"] = data["grid_position"].where(
        data["grid_position"].gt(0), data["field_size"].add(1)
    )
    data["post_quali_reached_q2"] = data["q2_seconds"].notna().astype("int8")
    data["post_quali_reached_q3"] = data["q3_seconds"].notna().astype("int8")
    for stage in (1, 2, 3):
        source = f"q{stage}_seconds"
        best = data.groupby(["season", "round"], observed=True)[source].transform("min")
        data[f"post_q{stage}_gap_pct"] = data[source].div(best).sub(1)
    groups = data.groupby(["season", "round", "constructor_id"], observed=True)
    count = groups["driver_id"].transform("size")
    teammate_position = (
        groups["quali_position"]
        .transform("sum")
        .sub(data["quali_position"])
        .div(count.sub(1).replace(0, np.nan))
    )
    data["post_quali_position_vs_teammate"] = data["quali_position"].sub(teammate_position)
    return data


def _feature_columns(frame: pd.DataFrame, *, post_quali: bool) -> list[str]:
    prefixes = (
        "driver_",
        "constructor_",
        "teammate_",
        "track_",
        "practice_",
        "openf1_",
        "static_",
    )
    excluded = (
        set(IDENTIFIERS)
        | set(TARGETS)
        | {
            "driver_dob",
            "driver_nationality",
            "constructor_nationality",
        }
    )
    columns = [
        column
        for column in frame
        if column not in excluded
        and (column.startswith(prefixes) or column in {"field_size", "season_round"})
    ]
    if post_quali:
        columns.extend(column for column in frame if column.startswith("post_"))
    return list(dict.fromkeys(columns))


def assert_no_leakage(pre: pd.DataFrame, post: pd.DataFrame, manifest: dict[str, Any]) -> None:
    pre_features = manifest["modes"]["pre_quali"]["features"]
    post_features = manifest["modes"]["post_quali"]["features"]
    forbidden = [
        column for column in pre_features if column.startswith(FORBIDDEN_PRE_QUALI_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Pre-qualifying leakage columns detected: {forbidden}")
    target_overlap = (set(pre_features) | set(post_features)) & set(TARGETS)
    if target_overlap:
        raise ValueError(f"Target columns exposed as features: {sorted(target_overlap)}")
    if pre.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("Pre-qualifying matrix contains duplicate race-driver rows")
    if post.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("Post-qualifying matrix contains duplicate race-driver rows")


def enrich_features(
    data: pd.DataFrame,
    paths: DataPaths,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the shared historical/live feature pipeline to a chronological base table."""
    data = _add_driver_history(data)
    data = data.merge(
        _team_race_table(data),
        on=["season", "round", "constructor_id"],
        how="left",
        validate="many_to_one",
    )
    data = _add_teammate_history(data)
    data = _add_track_history(data)
    data = _add_championship_context(data)
    data = _add_pit_history(data, paths)
    data = add_practice_features(
        data,
        paths.processed_f1db,
        paths.processed_openf1,
        schedule,
    )
    data = _add_post_quali_features(data)
    return data


def build_feature_matrices(paths: DataPaths) -> dict[str, Any]:
    schedule = read_partitioned_table(paths.processed_jolpica, "races")
    data = enrich_features(_base_table(paths), paths, schedule)

    pre_features = _feature_columns(data, post_quali=False)
    post_features = _feature_columns(data, post_quali=True)
    output_columns = [*IDENTIFIERS, *TARGETS]
    pre = data[[*output_columns, *pre_features]].copy()
    post = data[[*output_columns, *post_features]].copy()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": len(data),
        "races": int(data[["season", "round"]].drop_duplicates().shape[0]),
        "identifiers": IDENTIFIERS,
        "targets": TARGETS,
        "modes": {
            "pre_quali": {
                "features": pre_features,
                "cutoff": "strictly before qualifying starts",
            },
            "post_quali": {
                "features": post_features,
                "cutoff": "after qualifying classification and known starting grid",
            },
        },
        "leakage_policy": {
            "historical_results": "all rolling and expanding statistics use shift(1)",
            "practice": (
                "same-event practice is allowed only when the published session schedule "
                "places it before qualifying"
            ),
            "qualifying": "same-event qualifying and grid features are post_quali only",
            "outcomes": "target-race results are labels only",
        },
    }
    assert_no_leakage(pre, post, manifest)
    target_dir = paths.features
    atomic_parquet_write(pre, target_dir / "pre_quali.parquet")
    atomic_parquet_write(post, target_dir / "post_quali.parquet")
    atomic_json_write(target_dir / "manifest.json", manifest)
    return manifest
