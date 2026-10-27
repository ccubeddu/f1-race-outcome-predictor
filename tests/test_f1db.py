from __future__ import annotations

import zipfile

import pandas as pd

from f1predict.data.f1db import parse_practice_archive


def test_parse_practice_archive_normalizes_all_sessions(tmp_path) -> None:
    archive_path = tmp_path / "f1db.zip"
    driver = pd.DataFrame(
        [{"id": "example-driver", "fullName": "Example Driver", "abbreviation": "EXA"}]
    )
    race = pd.DataFrame(
        [
            {
                "id": 1,
                "freePractice1Date": "2024-02-29",
                "freePractice1Time": "11:30",
                "freePractice2Date": "2024-02-29",
                "freePractice2Time": "15:00",
                "freePractice3Date": "2024-03-01",
                "freePractice3Time": "12:30",
                "qualifyingDate": "2024-03-01",
                "qualifyingTime": "16:00",
            }
        ]
    )
    result = pd.DataFrame(
        [
            {
                "raceId": 1,
                "year": 2024,
                "round": 1,
                "positionDisplayOrder": 1,
                "positionText": "1",
                "driverNumber": 99,
                "driverId": "example-driver",
                "constructorId": "example-team",
                "engineManufacturerId": "example-engine",
                "tyreManufacturerId": "pirelli",
                "time": "1:30.123",
                "timeMillis": 90123,
                "gap": None,
                "gapMillis": 0,
                "laps": 20,
            }
        ]
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("f1db-drivers.csv", driver.to_csv(index=False))
        archive.writestr("f1db-races.csv", race.to_csv(index=False))
        for number in (1, 2, 3):
            archive.writestr(
                f"f1db-races-free-practice-{number}-results.csv",
                result.to_csv(index=False),
            )
    frame = parse_practice_archive(archive_path)
    assert frame["session"].tolist() == ["FP1", "FP2", "FP3"]
    assert frame["best_lap_seconds"].tolist() == [90.123, 90.123, 90.123]
    assert frame["driver_name"].unique().tolist() == ["Example Driver"]
