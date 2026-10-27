from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.io import read_partitioned_table
from f1predict.features.builder import (
    IDENTIFIERS,
    _base_table,
    _feature_columns,
    enrich_features,
)


@dataclass(frozen=True)
class RaceEvent:
    season: int
    round: int
    race_name: str
    race_date: pd.Timestamp
    circuit_id: str


def resolve_event(paths: DataPaths, season: int, race: str) -> RaceEvent:
    schedule = read_partitioned_table(paths.processed_jolpica, "races")
    schedule = schedule.loc[schedule["season"].eq(season)].copy()
    if schedule.empty:
        raise ValueError(f"No schedule for {season}; run `f1predict data sync` first.")
    query = race.strip()
    if query.isdigit():
        matches = schedule.loc[schedule["round"].eq(int(query))]
    else:
        needle = query.casefold()
        fields = ["race_name", "circuit_id", "circuit_name", "locality", "country"]
        match = pd.Series(False, index=schedule.index)
        for field in fields:
            match |= (
                schedule[field]
                .fillna("")
                .astype(str)
                .str.casefold()
                .str.contains(needle, regex=False)
            )
        matches = schedule.loc[match]
    if len(matches) != 1:
        choices = ", ".join(
            f"R{row['round']} {row['race_name']}" for _, row in matches.head(8).iterrows()
        )
        if matches.empty:
            choices = ", ".join(
                f"R{row['round']} {row['race_name']}" for _, row in schedule.iterrows()
            )
            raise ValueError(f"Race {race!r} was not found. Available: {choices}")
        raise ValueError(f"Race {race!r} is ambiguous. Matches: {choices}")
    row = matches.iloc[0]
    return RaceEvent(
        season=int(row["season"]),
        round=int(row["round"]),
        race_name=str(row["race_name"]),
        race_date=pd.to_datetime(row["race_date"]),
        circuit_id=str(row["circuit_id"]),
    )


def qualifying_available(paths: DataPaths, event: RaceEvent) -> bool:
    qualifying = read_partitioned_table(paths.processed_jolpica, "qualifying")
    rows = qualifying.loc[
        qualifying["season"].eq(event.season) & qualifying["round"].eq(event.round)
    ]
    return len(rows) >= 2 and rows["quali_position"].notna().mean() >= 0.5


def _target_rows(
    paths: DataPaths,
    historical: pd.DataFrame,
    schedule: pd.DataFrame,
    event: RaceEvent,
    mode: str,
) -> tuple[pd.DataFrame, str]:
    prior = historical.loc[historical["race_date"].lt(event.race_date)]
    if prior.empty:
        raise ValueError("No earlier race results are available to seed the target roster.")
    latest_date = prior["race_date"].max()
    latest_grid = prior.loc[prior["race_date"].eq(latest_date)].copy()
    qualifying = read_partitioned_table(paths.processed_jolpica, "qualifying")
    qualifying = qualifying.loc[
        qualifying["season"].eq(event.season) & qualifying["round"].eq(event.round)
    ].copy()
    if mode == "post_quali":
        if qualifying.empty:
            raise ValueError(
                "Post-qualifying mode requested, but no qualifying classification is synced."
            )
        roster = qualifying
        roster_source = "target qualifying classification; live grid defaults to qualifying order"
    else:
        # A practice classification includes reserve drivers, not confirmed race entrants.
        # Until a pre-Q entry-list source exists, explicitly expose this provisional roster.
        roster = latest_grid
        roster_source = (
            f"PROVISIONAL roster from latest completed race ({latest_date.date()}); "
            "entry changes need verification; practice participants are not a race entry list"
        )

    latest_by_id = prior.sort_values("race_date").drop_duplicates("driver_id", keep="last")
    latest_by_id = latest_by_id.set_index("driver_id")
    constructor_lookup = (
        prior.sort_values("race_date")
        .drop_duplicates("constructor_id", keep="last")
        .set_index("constructor_id")
    )
    event_row = schedule.loc[
        schedule["season"].eq(event.season) & schedule["round"].eq(event.round)
    ].iloc[0]
    rows: list[dict] = []
    for _, entrant in roster.iterrows():
        number = entrant.get("driver_number")
        driver_id = entrant.get("driver_id")
        if driver_id in latest_by_id.index:
            row = latest_by_id.loc[driver_id].to_dict()
        else:
            row = {column: np.nan for column in historical.columns}
        for column in roster.columns:
            if column in historical.columns and pd.notna(entrant[column]):
                row[column] = entrant[column]
        constructor_id = row.get("constructor_id")
        if constructor_id in constructor_lookup.index:
            constructor = constructor_lookup.loc[constructor_id]
            row["constructor_name"] = constructor["constructor_name"]
            row["constructor_nationality"] = constructor["constructor_nationality"]
        for column in schedule.columns:
            if column in historical.columns:
                row[column] = event_row[column]
        row["season"] = event.season
        row["round"] = event.round
        row["race_date"] = event.race_date
        row["race_name"] = event.race_name
        row["driver_name"] = entrant.get("driver_name", row.get("driver_name"))
        row["car_number"] = entrant.get("car_number", number)
        row["driver_number"] = row["car_number"]
        row["field_size"] = len(roster)
        row["season_round"] = event.round
        row["static_latitude"] = event_row["latitude"]
        row["static_longitude"] = event_row["longitude"]
        row["static_circuit_id"] = event.circuit_id
        row["static_country"] = event_row["country"]
        date_of_birth = pd.to_datetime(row.get("driver_dob"), errors="coerce")
        row["static_driver_age"] = (
            (event.race_date - date_of_birth).days / 365.2425 if pd.notna(date_of_birth) else np.nan
        )
        for column in (
            "finish_position",
            "finish_position_text",
            "finish_order",
            "points",
            "laps_completed",
            "status",
            "race_time_seconds",
            "fastest_lap_rank",
            "fastest_lap_number",
            "fastest_lap_seconds",
            "average_speed_kph",
            "completion_fraction",
            "target_finish_position",
            "target_win",
            "target_podium",
            "target_points",
            "target_dnf",
            "target_dns",
            "target_dsq",
            "target_classified",
        ):
            row[column] = np.nan
        if mode == "post_quali":
            row["grid_position"] = row.get("quali_position")
        else:
            for column in ("quali_position", "q1_seconds", "q2_seconds", "q3_seconds"):
                row[column] = np.nan
            row["grid_position"] = np.nan
        rows.append(row)
    target = pd.DataFrame(rows).reindex(columns=historical.columns)
    return target, roster_source


def build_live_race_features(
    paths: DataPaths,
    event: RaceEvent,
    mode: str,
) -> tuple[pd.DataFrame, str]:
    schedule = read_partitioned_table(paths.processed_jolpica, "races")
    historical = _base_table(paths)
    target, roster_source = _target_rows(paths, historical, schedule, event, mode)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries",
            category=FutureWarning,
        )
        combined = pd.concat([historical, target], ignore_index=True)
    combined = combined.sort_values(
        ["race_date", "season", "round", "driver_id"], ignore_index=True
    )
    enriched = enrich_features(combined, paths, schedule)
    live = enriched.loc[
        enriched["season"].eq(event.season) & enriched["round"].eq(event.round)
    ].copy()
    features = _feature_columns(enriched, post_quali=mode == "post_quali")
    return live[[*IDENTIFIERS, *features]], roster_source


def load_race_features(
    paths: DataPaths,
    event: RaceEvent,
    mode: str,
) -> tuple[pd.DataFrame, str, str]:
    matrix_path = paths.features / f"{mode}.parquet"
    matrix = pd.read_parquet(matrix_path)
    existing = matrix.loc[
        matrix["season"].eq(event.season) & matrix["round"].eq(event.round)
    ].copy()
    if not existing.empty:
        return existing, "historical feature matrix", "historical_replay"
    live, roster_source = build_live_race_features(paths, event, mode)
    return live, roster_source, "live_forecast"
