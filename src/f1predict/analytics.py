from __future__ import annotations

from typing import Any

import pandas as pd

from f1predict.config import DataPaths


def _match_driver(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    needle = query.strip().casefold()
    matches = pd.Series(False, index=frame.index)
    for column in ("driver_id", "driver_code", "driver_name"):
        matches |= (
            frame[column].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
        )
    identities = frame.loc[matches, ["driver_id", "driver_name"]].drop_duplicates()
    if len(identities) != 1:
        names = ", ".join(identities["driver_name"].astype(str))
        if identities.empty:
            available = ", ".join(sorted(frame["driver_name"].dropna().astype(str).unique()))
            raise ValueError(f"Driver {query!r} was not found. Available: {available}")
        raise ValueError(f"Driver {query!r} is ambiguous. Matches: {names}")
    return frame.loc[frame["driver_id"].eq(identities.iloc[0]["driver_id"])].copy()


def driver_qualifying_position_history(
    paths: DataPaths,
    *,
    driver: str,
    position: int,
    start_season: int | None = None,
    end_season: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if position < 1:
        raise ValueError("qualifying position must be positive")
    frame = pd.read_parquet(paths.features / "post_quali.parquet")
    driver_rows = _match_driver(frame, driver)
    if start_season is not None:
        driver_rows = driver_rows.loc[driver_rows["season"].ge(start_season)]
    if end_season is not None:
        driver_rows = driver_rows.loc[driver_rows["season"].le(end_season)]
    rows = driver_rows.loc[driver_rows["post_quali_position"].eq(position)].copy()
    if rows.empty:
        name = str(driver_rows["driver_name"].iloc[-1]) if len(driver_rows) else driver
        raise ValueError(f"No races found for {name} qualifying in P{position}.")
    rows["positions_gained"] = rows["post_quali_position"].sub(rows["target_finish_position"])
    classified = rows.loc[rows["target_dnf"].eq(0)]
    summary: dict[str, Any] = {
        "driver_id": str(rows["driver_id"].iloc[-1]),
        "driver_name": str(rows["driver_name"].iloc[-1]),
        "qualifying_position": position,
        "season_start": int(rows["season"].min()),
        "season_end": int(rows["season"].max()),
        "starts": len(rows),
        "wins": int(rows["target_win"].sum()),
        "win_rate": float(rows["target_win"].mean()),
        "podiums": int(rows["target_podium"].sum()),
        "podium_rate": float(rows["target_podium"].mean()),
        "points_finishes": int(rows["target_points"].sum()),
        "points_rate": float(rows["target_points"].mean()),
        "dnfs": int(rows["target_dnf"].sum()),
        "dnf_rate": float(rows["target_dnf"].mean()),
        "mean_finish": float(rows["target_finish_position"].mean()),
        "median_finish": float(rows["target_finish_position"].median()),
        "mean_positions_gained": float(rows["positions_gained"].mean()),
        "classified_mean_finish": float(classified["target_finish_position"].mean()),
        "classified_mean_positions_gained": float(classified["positions_gained"].mean()),
    }
    history = rows[
        [
            "season",
            "round",
            "race_date",
            "race_name",
            "constructor_name",
            "target_finish_position",
            "positions_gained",
            "target_dnf",
        ]
    ].sort_values("race_date", ascending=False, ignore_index=True)
    return summary, history
