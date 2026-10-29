import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write
from f1predict.evaluation import rating_study as study
from f1predict.evaluation.grid_simulation import position_matrix, prediction_frame
from f1predict.features.ratings import load_event_layouts


def test_layout_dates_fail_closed_and_actual_race_distance_is_not_a_feature(tmp_path: Path):
    raw = pd.DataFrame(
        [
            dict(
                year=2024,
                round=1,
                date="2024-03-02",
                circuitLayoutId="layout1",
                circuitType="RACE",
                direction="CLOCKWISE",
                courseLength=5.0,
                turns=15,
                distance=9999,
            ),
            dict(
                year=2024,
                round=2,
                date="2024-03-09",
                circuitLayoutId="layout1",
                circuitType="RACE",
                direction="CLOCKWISE",
                courseLength=5.0,
                turns=15,
                distance=1,
            ),
        ]
    )
    archive_path = tmp_path / "f1db.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("f1db-races.csv", raw.to_csv(index=False))
    races = pd.DataFrame(
        [
            dict(season=2024, round=1, race_date="2024-03-02"),
            dict(season=2024, round=2, race_date="2024-03-10"),
        ]
    )
    layouts = load_event_layouts(archive_path, races)
    assert layouts.rating_layout_available.tolist() == [1, 0]
    assert np.isnan(layouts.loc[1, "rating_layout_length_km"])
    assert layouts.loc[1, "layout_bucket"] == "unverified_2024_2"
    assert "distance" not in layouts
    assert len(layouts.attrs["layout_exclusions"]) == 1


def test_prediction_audit_checks_identical_grid_support_and_probability_identities():
    drivers = pd.DataFrame(
        {
            "season": [2024] * 4,
            "round": [1] * 4,
            "driver_id": list("abcd"),
            "race_date": ["2024-03-02"] * 4,
        }
    )
    matrix = position_matrix(np.arange(4), np.full(4, 0.1), simulations=1024)
    forecasts = prediction_frame(drivers, matrix, np.full(4, 0.1)).assign(training_end="2024-02-01")
    data = pd.concat(
        [forecasts.assign(candidate=c) for c in ("direct_ensemble", "ratings_ensemble")]
    )
    assert study.audit_rating_predictions(data)["passed"]
    bad = data.copy()
    bad.iloc[0, bad.columns.get_loc("p_win")] = 0.99
    assert not study.audit_rating_predictions(bad)["passed"]
    late = data.assign(training_end="2024-03-02")
    assert not study.audit_rating_predictions(late)["passed"]


def test_rating_walk_forward_is_resumable_and_predicts_without_current_labels(
    tmp_path, monkeypatch
):
    stage, production = DataPaths(tmp_path / "study/data"), DataPaths(tmp_path / "production")
    rows = []
    for r in range(52):
        date = pd.date_range("2010-01-01", periods=52, freq="30D")[r]
        if r >= 50:
            date = pd.date_range("2023-01-01", periods=2, freq="30D")[r - 50]
        for d in range(4):
            rows.append(
                dict(
                    season=2023 if r >= 50 else 2010 + r // 12,
                    round=r + 1,
                    race_date=date,
                    race_name=f"Race {r}",
                    driver_id=f"d{d}",
                    driver_name=f"D{d}",
                    constructor_name="Team",
                    field_size=4,
                    driver_form=4 - d,
                    label_quali_position=d + 1,
                    target_finish_position=d + 1,
                    target_win=int(d == 0),
                    target_podium=int(d < 3),
                    target_points=1,
                    target_dnf=int(d == 3),
                    rating_global=0.0,
                    rating_context_track=0.0,
                    rel_driver_dnf_rate=0.1,
                    asof_race_date=pd.NaT,
                )
            )
    data = pd.DataFrame(rows)
    base = ["field_size", "driver_form"]
    features = {
        "direct": base,
        "global": base + ["rating_global"],
        "ratings": base + ["rating_global", "rating_context_track"],
        "reliability": base + ["rating_global", "rating_context_track", "rel_driver_dnf_rate"],
    }
    manifest = {
        "families": {
            "global": ["rating_global"],
            "context": ["rating_context_track"],
            "reliability": ["rel_driver_dnf_rate"],
        }
    }
    atomic_json_write(
        stage.root / "baseline_snapshot.json",
        {"production_root": str(production.root.resolve()), "production_fingerprints": {}},
    )
    for path in (
        stage.features / "pre_quali.parquet",
        stage.features / "post_quali.parquet",
        stage.root.parent / "rating_features/ratings.parquet",
    ):
        atomic_parquet_write(data, path)
    atomic_json_write(stage.root.parent / "rating_features/manifest.json", manifest)
    monkeypatch.setattr(study, "load_rating_data", lambda *args: (data, features, manifest))
    fits = []

    class Fake:
        def predict_strength(self, inputs):
            assert not any(c.startswith(("target_", "label_", "post_")) for c in inputs)
            return inputs.driver_form.to_numpy(dtype=float)

    def fit(training, features, kinds, **kwargs):
        fits.append(training.race_date.max())
        return {k: Fake() for k in kinds}

    monkeypatch.setattr(study, "fit_finish_candidates", fit)
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
        "summarize_rating_study",
        lambda predictions, output, params, manifest: {
            "rows": len(predictions),
            "run_id": params["run_id"],
        },
    )
    result = study.run_rating_study(
        stage,
        production,
        tmp_path / "output",
        start_season=2023,
        end_season=2023,
        warmup_races=5,
        simulations=256,
    )
    assert result["rows"] == 2 * 4 * 7
    assert len(fits) == 7 * 3
    assert not production.root.exists()
    fits.clear()
    replay = study.run_rating_study(
        stage,
        production,
        tmp_path / "output",
        start_season=2023,
        end_season=2023,
        warmup_races=5,
        simulations=256,
    )
    assert result == replay
    assert not fits
