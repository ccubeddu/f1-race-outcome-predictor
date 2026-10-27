from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TableContract:
    required: tuple[str, ...]
    key: tuple[str, ...]


CONTRACTS: dict[str, TableContract] = {
    "races": TableContract(
        required=("season", "round", "race_name", "race_date", "circuit_id"),
        key=("season", "round"),
    ),
    "results": TableContract(
        required=(
            "season",
            "round",
            "driver_id",
            "constructor_id",
            "finish_position",
            "status",
        ),
        key=("season", "round", "driver_id"),
    ),
    "qualifying": TableContract(
        required=("season", "round", "driver_id", "constructor_id", "quali_position"),
        key=("season", "round", "driver_id"),
    ),
    "sprints": TableContract(
        required=("season", "round", "driver_id", "constructor_id", "finish_position"),
        key=("season", "round", "driver_id"),
    ),
    "pit_stops": TableContract(
        required=("season", "round", "driver_id", "stop", "lap"),
        key=("season", "round", "driver_id", "stop"),
    ),
    "f1db_practice_results": TableContract(
        required=(
            "season",
            "round",
            "session",
            "driver_id",
            "position",
        ),
        key=("season", "round", "session", "driver_id"),
    ),
    "openf1_practice_sessions": TableContract(
        required=("season", "round", "session_key", "session_name", "date_end"),
        key=("session_key",),
    ),
    "openf1_practice_results": TableContract(
        required=("season", "round", "session_key", "driver_number", "position"),
        key=("session_key", "driver_number"),
    ),
    "openf1_practice_laps": TableContract(
        required=("season", "round", "session_key", "driver_number", "lap_number"),
        key=("session_key", "driver_number", "lap_number"),
    ),
    "openf1_practice_stints": TableContract(
        required=("season", "round", "session_key", "driver_number", "stint_number"),
        key=("session_key", "driver_number", "stint_number"),
    ),
    "openf1_practice_weather": TableContract(
        required=("season", "round", "session_key", "date"),
        key=("session_key", "date"),
    ),
}


def validate_table(name: str, frame: pd.DataFrame) -> None:
    contract = CONTRACTS[name]
    if frame.empty:
        return
    missing = sorted(set(contract.required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {', '.join(missing)}")
    duplicate_count = int(frame.duplicated(list(contract.key)).sum())
    if duplicate_count:
        raise ValueError(f"{name}: {duplicate_count} duplicate primary keys for {contract.key}")
    null_keys = int(frame[list(contract.key)].isna().any(axis=1).sum())
    if null_keys:
        raise ValueError(f"{name}: {null_keys} rows contain null primary-key values")
