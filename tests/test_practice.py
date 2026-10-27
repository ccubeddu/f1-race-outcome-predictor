from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1predict.data.io import atomic_parquet_write
from f1predict.features.practice import (
    _eligible,
    _long_run_features,
    _scheduled_pre_quali_sessions,
    _wide,
    add_practice_features,
    match_practice_drivers,
)


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                season=2024,
                round=1,
                fp1_date="2024-03-01",
                fp1_time="10:00:00Z",
                fp2_date="2024-03-01",
                fp2_time="14:00:00Z",
                fp3_date="2024-03-02",
                fp3_time="10:00:00Z",
                qualifying_date="2024-03-02",
                qualifying_time="14:00:00Z",
            )
        ]
    )


def test_identity_not_car_number_and_conflicting_evidence_rejected() -> None:
    roster = pd.DataFrame(
        [
            dict(
                season=2024,
                round=1,
                driver_number=1,
                driver_id="max",
                driver_code="VER",
                driver_name="Max Verstappen",
            ),
            dict(
                season=2024,
                round=1,
                driver_number=44,
                driver_id="lewis",
                driver_code="HAM",
                driver_name="Lewis Hamilton",
            ),
        ]
    )
    practice = pd.DataFrame(
        [
            dict(season=2024, round=1, driver_number=1, driver_code="RES", driver_name="Reserve"),
            dict(
                season=2024,
                round=1,
                driver_number=99,
                driver_code="VER",
                driver_name="MAX VERSTAPPEN",
            ),
            dict(
                season=2024,
                round=1,
                driver_number=1,
                driver_code="VER",
                driver_name="Lewis Hamilton",
            ),
            dict(season=2024, round=1, driver_number=44, driver_code=None, driver_name=None),
        ]
    )
    ids = match_practice_drivers(practice, roster)["driver_id"]
    assert ids.isna().tolist() == [True, False, True, True]
    assert ids.iloc[1] == "max"


def test_same_day_start_does_not_prove_completion_and_asof_applies() -> None:
    records = pd.DataFrame(
        [dict(season=2024, round=1, session="FP3", date_end="2024-03-02T11:00:00Z")]
    )
    assert (
        not _scheduled_pre_quali_sessions(schedule())
        .set_index("session")
        .loc["FP3", "is_pre_quali"]
    )
    before_end = _scheduled_pre_quali_sessions(
        schedule(), records, as_of=pd.Timestamp("2024-03-02T10:30:00Z")
    )
    assert not before_end.set_index("session").loc["FP3", "is_pre_quali"]
    completed = _scheduled_pre_quali_sessions(
        schedule(), records, as_of=pd.Timestamp("2024-03-02T13:00:00Z")
    )
    assert completed.set_index("session").loc["FP3", "is_pre_quali"]
    records["date_end"] = "2024-03-02T14:01:00Z"
    assert (
        not _scheduled_pre_quali_sessions(schedule(), records)
        .set_index("session")
        .loc["FP3", "is_pre_quali"]
    )


def test_all_tables_and_samples_share_cutoff() -> None:
    eligibility = _scheduled_pre_quali_sessions(schedule())
    for table in ("laps", "stints", "weather"):
        frame = pd.DataFrame(
            [
                dict(season=2024, round=1, session="FP1", date="2024-03-01T11:00:00Z"),
                dict(season=2024, round=1, session="FP3", date="2024-03-02T11:00:00Z"),
                dict(season=2024, round=1, session="FP1", date="2024-03-02T15:00:00Z"),
            ]
        )
        assert len(_eligible(frame, eligibility)) == 1, table


def test_lap_finishing_after_cutoff_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            dict(
                season=2024,
                round=1,
                session="FP1",
                date_start="2024-03-02T13:59:30Z",
                lap_duration=90,
            )
        ]
    )
    assert _eligible(frame, _scheduled_pre_quali_sessions(schedule())).empty


def test_pivot_does_not_silently_choose_duplicate_identity() -> None:
    frame = pd.DataFrame([dict(season=2024, round=1, session="FP1", driver_id="max", x=1)] * 2)
    with pytest.raises(ValueError, match="Duplicate practice"):
        _wide(frame, ["season", "round", "driver_id"], ["x"], prefix="practice")


def test_long_run_compound_age_peers_and_minimum_laps() -> None:
    laps, stints = [], []
    for driver, duration in ((1, 90), (2, 91), (3, 92), (4, 85)):
        common = dict(season=2024, round=1, session="FP2", session_key=10, driver_number=driver)
        stints.append(
            dict(
                **common,
                stint_number=1,
                lap_start=1,
                lap_end=9,
                compound="SOFT" if driver == 4 else "MEDIUM",
                tyre_age_at_start=0,
            )
        )
        laps.extend(dict(**common, lap_number=lap, lap_duration=duration) for lap in range(1, 10))
    result = _long_run_features(pd.DataFrame(laps), pd.DataFrame(stints)).set_index("driver_number")
    assert result.loc[1, "detail_long_run_clean_laps"] == 7  # boundaries excluded
    assert result.loc[1, "detail_tyre_adjusted_gap_pct"] == pytest.approx(90 / 91 - 1)
    assert np.isnan(result.loc[4, "detail_tyre_adjusted_gap_pct"])  # no soft peers
    short = pd.DataFrame(laps).loc[lambda df: df["lap_number"].le(4)]
    assert _long_run_features(short, pd.DataFrame(stints)).empty
    missing_age = pd.DataFrame(stints).assign(tyre_age_at_start=np.nan)
    assert _long_run_features(pd.DataFrame(laps), missing_age).empty
    short_peers = pd.DataFrame(laps).loc[
        lambda df: df["driver_number"].eq(1) | df["lap_number"].le(4)
    ]
    result = _long_run_features(short_peers, pd.DataFrame(stints))
    assert result["detail_tyre_adjusted_gap_pct"].isna().all()


def test_add_features_does_not_attribute_reserve_to_race_driver(tmp_path: Path) -> None:
    base = pd.DataFrame(
        [
            dict(
                season=2024,
                round=1,
                driver_id="max",
                driver_number=1,
                driver_code="VER",
                driver_name="Max Verstappen",
            )
        ]
    )
    practice = pd.DataFrame(
        [
            dict(
                season=2024,
                round=1,
                session="FP1",
                driver_id="reserve",
                driver_number=1,
                driver_code="RES",
                driver_name="Reserve",
                best_lap_seconds=80,
                position=1,
                laps_completed=30,
            )
        ]
    )
    atomic_parquet_write(practice, tmp_path / "f1db/practice_results/season=2024/data.parquet")
    result = add_practice_features(base, tmp_path / "f1db", tmp_path / "openf1", schedule())
    assert result.loc[0, "practice_data_available"] == 0
    assert result.loc[0, "openf1_detail_available"] == 0
