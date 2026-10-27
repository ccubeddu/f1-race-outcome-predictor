from pathlib import Path

import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.io import atomic_parquet_write
from f1predict.features.live import RaceEvent, _target_rows


def test_live_roster_never_infers_reserve_entry_or_identity_from_number(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    history = pd.DataFrame(
        [
            dict(
                season=2024,
                round=1,
                race_date=pd.Timestamp("2024-03-01"),
                driver_id=driver,
                driver_number=number,
                car_number=number,
                driver_code=driver.upper(),
                driver_name=driver,
                driver_dob=dob,
                constructor_id="team",
                constructor_name="Team",
                constructor_nationality="British",
                static_driver_age=20.0,
            )
            for driver, number, dob in (("one", 1, "1990-01-01"), ("two", 2, "2000-01-01"))
        ]
    )
    event = RaceEvent(2024, 2, "Race", pd.Timestamp("2024-03-10"), "track")
    schedule = pd.DataFrame(
        [
            dict(
                season=2024,
                round=2,
                race_date="2024-03-10",
                latitude=1,
                longitude=2,
                country="Country",
            )
        ]
    )
    qualifying = pd.DataFrame(
        [
            dict(
                season=2024,
                round=2,
                driver_id="one",
                driver_code="ONE",
                driver_number=2,
                car_number=2,
                quali_position=1,
            )
        ]
    )
    atomic_parquet_write(
        qualifying, paths.processed_jolpica / "qualifying/season=2024/data.parquet"
    )
    # A reserve borrowed driver one's number. This is NOT evidence they will race.
    practice = pd.DataFrame(
        [dict(season=2024, round=2, driver_number=1, driver_id="reserve", driver_code="RES")]
    )
    atomic_parquet_write(
        practice, paths.processed_f1db / "practice_results/season=2024/data.parquet"
    )
    post, _ = _target_rows(paths, history, schedule, event, "post_quali")
    assert post.loc[0, "driver_dob"] == "1990-01-01"  # driver one, not car two
    pre, source = _target_rows(paths, history, schedule, event, "pre_quali")
    assert set(pre["driver_id"]) == {"one", "two"}
    assert "PROVISIONAL" in source
