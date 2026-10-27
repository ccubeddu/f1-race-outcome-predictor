from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from f1predict.config import DataPaths
from f1predict.data.contracts import CONTRACTS
from f1predict.data.history import audit_history_expansion, prepare_history_stage
from f1predict.data.io import atomic_json_write, atomic_parquet_write
from f1predict.data.pipeline import sync_f1db, sync_jolpica


def baseline(tmp_path) -> DataPaths:
    paths = DataPaths(tmp_path / "production")
    paths.create()
    result = pd.DataFrame(
        [
            {
                "season": 2018,
                "round": 1,
                "driver_id": "driver",
                "constructor_id": "team",
                "finish_position": 1,
                "status": "Finished",
                "race_date": "2018-03-25",
            }
        ]
    )
    atomic_parquet_write(result, paths.processed_jolpica / "results/season=2018/data.parquet")
    paths.models.mkdir()
    (paths.models / "model.joblib").write_bytes(b"unchanged reference model")
    atomic_json_write(
        paths.catalog,
        {
            "sources": {
                "jolpica": {"rows": {"2018": {"races": 1, "results": 1}}},
                "f1db": {"practice_rows": {"2018": 3}},
            }
        },
    )
    return paths


def test_staging_preserves_production_and_resumes_without_copying_models(tmp_path) -> None:
    production = baseline(tmp_path)
    stage = DataPaths(tmp_path / "stage")
    manifest = prepare_history_stage(production, stage)
    assert manifest["baseline_races"] == 1
    assert not stage.models.exists()
    assert (stage.processed_jolpica / "results/season=2018/data.parquet").exists()
    assert prepare_history_stage(production, stage) == manifest
    (production.models / "model.joblib").write_bytes(b"new production model")
    with pytest.raises(ValueError, match="Production data changed"):
        prepare_history_stage(production, stage)


def test_staging_rejects_nested_or_unowned_nonempty_directory(tmp_path) -> None:
    production = baseline(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        prepare_history_stage(production, DataPaths(production.root / "stage"))
    stage = tmp_path / "occupied"
    stage.mkdir()
    (stage / "user.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="nonempty"):
        prepare_history_stage(production, DataPaths(stage))
    assert (stage / "user.txt").read_text() == "preserve me"


def test_incremental_sync_preserves_prior_season_catalogs(tmp_path, monkeypatch) -> None:
    paths = baseline(tmp_path)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("f1predict.data.pipeline.JolpicaClient", FakeClient)
    monkeypatch.setattr(
        "f1predict.data.pipeline.download_season",
        lambda client, year: {
            name: pd.DataFrame(columns=CONTRACTS[name].required)
            for name in ("races", "results", "qualifying", "sprints", "pit_stops")
        },
    )
    sync_jolpica(paths, 2010, 2010)
    catalog = json.loads(paths.catalog.read_text())
    assert set(catalog["sources"]["jolpica"]["rows"]) == {"2010", "2018"}
    assert catalog["sources"]["jolpica"]["season_start"] == 2010
    assert catalog["sources"]["jolpica"]["season_end"] == 2018
    monkeypatch.setattr(
        "f1predict.data.pipeline.f1db_practice_data",
        lambda *args, **kwargs: pd.DataFrame(
            [{"season": 2010, "round": 1, "session": "FP1", "driver_id": "driver", "position": 1}]
        ),
    )
    sync_f1db(paths, 2010, 2010)
    catalog = json.loads(paths.catalog.read_text())
    assert catalog["sources"]["f1db"]["practice_rows"] == {"2010": 1, "2018": 3}


def test_expansion_audit_flags_unknown_timing_and_substitute_driver(tmp_path) -> None:
    production = baseline(tmp_path)
    stage = DataPaths(tmp_path / "stage")
    prepare_history_stage(production, stage)
    atomic_parquet_write(
        pd.DataFrame(
            [
                {
                    "season": 2010,
                    "round": 1,
                    "driver_id": "driver",
                    "constructor_id": "team",
                    "finish_position": 1,
                    "status": "Finished",
                    "race_date": "2010-03-14",
                    "car_number": 1,
                    "driver_number": 1,
                    "given_name": "Race",
                    "family_name": "Driver",
                    "grid_position": 1,
                }
            ]
        ),
        stage.processed_jolpica / "results/season=2010/data.parquet",
    )
    race = {
        "season": 2010,
        "round": 1,
        "race_name": "Test GP",
        "race_date": "2010-03-14",
        "circuit_id": "test",
        "qualifying_date": None,
        "qualifying_time": None,
    }
    race.update(
        {
            f"{session}_{field}": None
            for session in ("fp1", "fp2", "fp3")
            for field in ("date", "time")
        }
    )
    atomic_parquet_write(
        pd.DataFrame([race]), stage.processed_jolpica / "races/season=2010/data.parquet"
    )
    atomic_parquet_write(
        pd.DataFrame(
            [
                {
                    "season": 2010,
                    "round": 1,
                    "driver_id": "driver",
                    "constructor_id": "team",
                    "quali_position": 1,
                }
            ]
        ),
        stage.processed_jolpica / "qualifying/season=2010/data.parquet",
    )
    atomic_parquet_write(
        pd.DataFrame(columns=CONTRACTS["pit_stops"].required),
        stage.processed_jolpica / "pit_stops/season=2010/data.parquet",
    )
    atomic_parquet_write(
        pd.DataFrame(
            [
                {
                    "season": 2010,
                    "round": 1,
                    "session": "FP1",
                    "driver_id": "substitute",
                    "driver_number": 1,
                    "driver_name": "Reserve Driver",
                    "position": 1,
                    "best_lap_seconds": 90.0,
                    "session_date": None,
                    "session_time": None,
                }
            ]
        ),
        stage.processed_f1db / "practice_results/season=2010/data.parquet",
    )
    with zipfile.ZipFile(stage.raw_f1db / "f1db-csv.zip", "w") as archive:
        archive.writestr("f1db-races-race-results.csv", "year,round\n2010,1\n")
    report, coverage = audit_history_expansion(production, stage, start=2010, end=2010)
    assert report["production_unchanged"]
    assert not report["errors"]
    assert report["added_races"] == 1
    assert report["practice"]["same_number_different_driver_rows"] == 1
    assert report["practice"]["timing_status_counts"] == {"missing_schedule_dates": 1}
    assert coverage.loc[coverage.season.eq(2010), "qualifying_match_rate"].iloc[0] == 1.0
