from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write
from f1predict.evaluation import qualifying_study as study


def test_chronological_study_is_label_safe_resumable_and_non_promoting(tmp_path, monkeypatch):
    production, stage = DataPaths(tmp_path / "production"), DataPaths(tmp_path / "stage/data")
    atomic_json_write(
        stage.root / "baseline_snapshot.json",
        {"production_root": str(production.root.resolve()), "production_fingerprints": {}},
    )
    rows = []
    for race in range(52):
        date = pd.date_range("2010-01-01", periods=52, freq="30D")[race]
        if race >= 50:
            date = pd.date_range("2023-01-01", periods=2, freq="30D")[race - 50]
        for driver in range(4):
            rows.append(
                dict(
                    season=2023 if race >= 50 else 2010 + race // 12,
                    round=race + 1,
                    driver_id=f"d{driver}",
                    constructor_id=f"t{driver // 2}",
                    race_date=date,
                    race_name=f"Race {race}",
                    driver_name=f"D{driver}",
                    constructor_name="Team",
                    field_size=4,
                    driver_form=float(4 - driver),
                    target_finish_position=driver + 1,
                    target_win=int(driver == 0),
                    target_podium=int(driver < 3),
                    target_points=1,
                    target_dnf=int(driver == 3),
                    post_quali_position=driver + 1,
                )
            )
    data = pd.DataFrame(rows)
    atomic_parquet_write(
        data.drop(columns="post_quali_position"), stage.features / "pre_quali.parquet"
    )
    atomic_parquet_write(data, stage.features / "post_quali.parquet")
    atomic_json_write(
        stage.features / "manifest.json",
        {"schema_version": 2, "modes": {"pre_quali": {"features": ["field_size", "driver_form"]}}},
    )
    training_cutoffs = []

    class FakeModel:
        def __init__(self, historical):
            self.training_end = historical["race_date"].max()

        def predict_strength(self, inputs):
            assert self.training_end < inputs["race_date"].min()
            assert not any(c.startswith(("target_", "label_", "post_")) for c in inputs)
            if "scenario_quali_position" in inputs:
                return -inputs["scenario_quali_position"].to_numpy()
            return inputs["driver_form"].to_numpy()

    def fit_q(training, features, **kwargs):
        training_cutoffs.append(training["race_date"].max())
        assert not any(c.startswith(("post_", "label_", "scenario_", "target_")) for c in features)
        return FakeModel(training)

    def fit_race(training, features, kinds, **kwargs):
        assert not any(c.startswith(("post_", "label_", "target_")) for c in features)
        return {kind: FakeModel(training) for kind in kinds}

    monkeypatch.setattr(study, "fit_qualifying_model", fit_q)
    monkeypatch.setattr(study, "fit_finish_candidates", fit_race)
    monkeypatch.setattr(
        study,
        "_calibrated_dnf",
        lambda training, test, *args: (np.full(len(test), 0.1), np.full(len(test), 0.1)),
    )
    monkeypatch.setattr(
        study,
        "_importance",
        lambda _: pd.DataFrame({"feature": ["driver_form"], "gain": [1.0], "importance": [1.0]}),
    )
    monkeypatch.setattr(
        study,
        "summarize_study",
        lambda predictions, qualifying, output, params: {
            "rows": len(predictions),
            "qualifying_rows": len(qualifying),
            "run_id": params["run_id"],
        },
    )
    output = tmp_path / "report"
    result = study.run_qualifying_study(
        stage,
        production,
        output,
        start_season=2023,
        end_season=2023,
        warmup_races=5,
        grid_scenarios=32,
        simulations=256,
    )
    assert result["rows"] == 2 * 4 * 5
    assert result["qualifying_rows"] == 8
    assert len(training_cutoffs) == 7
    cutoff_table = pd.read_csv(output / "cutoffs.csv")
    assert (
        pd.to_datetime(cutoff_table.training_end) < pd.to_datetime(cutoff_table.race_date)
    ).all()
    assert not production.root.exists()

    def should_not_train(*args, **kwargs):
        raise AssertionError("Cached fold unexpectedly retrained")

    monkeypatch.setattr(study, "fit_qualifying_model", should_not_train)
    second = study.run_qualifying_study(
        stage,
        production,
        output,
        start_season=2023,
        end_season=2023,
        warmup_races=5,
        grid_scenarios=32,
        simulations=256,
    )
    assert result == second


def test_study_refuses_production_as_staging(tmp_path: Path):
    paths = DataPaths(tmp_path)
    with pytest.raises(ValueError, match="isolated"):
        study.run_qualifying_study(paths, paths, tmp_path / "output")
