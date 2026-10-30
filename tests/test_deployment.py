import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write
from f1predict.evaluation import deployment_study as study
from f1predict.models import deployment as recipe
from f1predict.models.candidates import FinishCandidate, _weighted_order
from f1predict.models.preprocessing import build_preprocessor, clean_features


class ToyScore:
    def __init__(self, sign=1):
        self.sign = sign

    def predict(self, matrix):
        return self.sign * np.asarray(matrix)[:, 0]


class ToyDnf:
    def fit(self, matrix, target, sample_weight=None):
        return self

    def predict_proba(self, matrix):
        p = np.full(len(matrix), 0.15)
        return np.column_stack([1 - p, p])


def toy_data():
    dates = [
        *pd.date_range("2020-01-01", periods=50, freq="5D"),
        *pd.date_range("2022-01-01", periods=5, freq="5D"),
        *pd.to_datetime([f"{year}-03-01" for year in range(2023, 2027)]),
    ]
    rows = []
    for index, date in enumerate(dates):
        for driver in range(4):
            rows.append(
                {
                    "season": date.year,
                    "round": index + 1,
                    "race_date": date,
                    "race_name": f"Race {index}",
                    "driver_id": f"d{driver}",
                    "driver_name": f"Driver {driver}",
                    "constructor_name": "Team",
                    "driver_form": 4 - driver,
                    "field_size": 4,
                    "post_quali_position": driver + 1,
                    "target_finish_position": driver + 1,
                    "target_win": int(driver == 0),
                    "target_podium": int(driver < 3),
                    "target_points": 1,
                    "target_dnf": int(driver == 3),
                }
            )
    return pd.DataFrame(rows)


def test_weighted_order_keeps_weights_attached_to_input_rows():
    frame = toy_data().iloc[[6, 0, 5, 2]].copy()
    weights = np.array([60, 10, 50, 20.0])
    expected = dict(zip(zip(frame.driver_id, frame.race_date, strict=True), weights, strict=True))
    ordered, actual = _weighted_order(frame, weights)
    for row, weight in zip(ordered.itertuples(), actual, strict=True):
        assert expected[(row.driver_id, row.race_date)] == weight
    with pytest.raises(ValueError):
        _weighted_order(frame, np.array([1, -1, 2, 3]))


def test_training_window_weights_and_future_results():
    data = toy_data()
    config = recipe.DeploymentConfig(2018, 730.5)
    rows, weights = recipe.training_rows(data, pd.Timestamp("2024-03-01"), config)
    assert rows.race_date.max() < pd.Timestamp("2024-03-01")
    assert weights.mean() == pytest.approx(1)
    assert np.all(np.diff(weights) >= 0)
    for indices in rows.groupby(["season", "round"]).indices.values():
        assert np.ptp(weights[indices]) == 0
    changed = data.copy()
    changed.loc[changed.season.ge(2024), "target_finish_position"] = 999
    other, other_weights = recipe.training_rows(changed, pd.Timestamp("2024-03-01"), config)
    pd.testing.assert_frame_equal(rows, other)
    np.testing.assert_array_equal(weights, other_weights)


def test_calibration_rejects_current_future_duplicate_and_unordered_records():
    config = recipe.DeploymentConfig()
    for dates in (
        ["2024-03-01"],
        ["2025-01-01"],
        ["2023-01-01", "2023-01-01"],
        ["2023-02-01", "2023-01-01"],
    ):
        with pytest.raises(ValueError, match="strictly earlier"):
            recipe.calibration_parameters(
                [{"race_date": d} for d in dates], pd.Timestamp("2024-03-01"), config, "pre_quali"
            )


def test_calibration_window_drops_old_records():
    history = [
        dict(
            race_date=str(date),
            raw_dnf=[0.2, 0.2],
            actual_dnf=[0, 1],
            losses={kind: list(range(7)) for kind in recipe.ENSEMBLE_WEIGHTS["pre_quali"]},
        )
        for date in pd.date_range("2020-01-01", periods=31)
    ]
    _, temperatures, metadata = recipe.calibration_parameters(
        history, pd.Timestamp("2024-01-01"), recipe.DeploymentConfig(), "pre_quali"
    )
    assert metadata["races"] == 30
    assert metadata["start"].startswith("2020-01-02")
    assert set(temperatures.values()) == {0.3}


def test_selection_uses_all_regression_guards_and_baseline_fallback():
    def metric(loss, win=1, rank=0.6):
        return dict(
            position_log_loss=loss,
            winner_log_loss=win,
            spearman=rank,
            top3_accuracy=0.6,
            top10_accuracy=0.8,
            probabilities={"dnf": {"brier": 0.1}},
        )

    baseline = metric(2.7)
    values = {
        study.BASELINE: baseline,
        "good": metric(2.6),
        "bad_win": metric(2.0, win=1.5),
        "bad_rank": metric(1.9, rank=0.5),
    }
    assert study.select_configuration(values)["candidate"] == "good"
    values["good"] = metric(2.8)
    assert study.select_configuration(values)["candidate"] == study.BASELINE


def test_full_study_resumes_and_saved_bundle_replays_without_target_labels(tmp_path, monkeypatch):
    stage, production = DataPaths(tmp_path / "study/data"), DataPaths(tmp_path / "production")
    output = tmp_path / "outputs"
    data = toy_data()
    frames = {mode: data.copy() for mode in study.MODES}
    contracts = {
        mode: {"base": ["driver_form", "field_size"], "ratings": ["driver_form", "field_size"]}
        for mode in study.MODES
    }
    configs = [recipe.DeploymentConfig(), recipe.DeploymentConfig(2010, 730.5)]
    monkeypatch.setattr(study, "_load_inputs", lambda stage: (frames, contracts, {}))
    monkeypatch.setattr(study, "default_configurations", lambda: configs)
    monkeypatch.setattr(
        study,
        "_importance",
        lambda model: pd.DataFrame(
            {"feature": ["driver_form"], "gain": [1.0], "importance": [1.0]}
        ),
    )
    from f1predict.evaluation import deployment_report

    monkeypatch.setattr(deployment_report, "write_deployment_report", lambda *args: None)
    for mode in study.MODES:
        atomic_parquet_write(data, stage.features / f"{mode}.parquet")
    feature_manifest = {
        "schema_version": 2,
        "modes": {
            mode: {"features": contracts[mode]["base"], "cutoff": mode} for mode in study.MODES
        },
    }
    atomic_json_write(stage.features / "manifest.json", feature_manifest)
    atomic_parquet_write(
        pd.DataFrame({"value": [1]}), stage.root.parent / "rating_features/ratings.parquet"
    )
    atomic_json_write(stage.root.parent / "rating_features/manifest.json", {})
    atomic_json_write(
        stage.root / "baseline_snapshot.json",
        {"production_root": str(production.root.resolve()), "production_fingerprints": {}},
    )
    calls = []

    def fit_models(training, features, kinds, *, profile, sample_weight, n_jobs):
        assert profile == "full" and n_jobs == 1
        if sample_weight is not None:
            assert sample_weight.mean() == pytest.approx(1)
        calls.append(training.race_date.max())
        prep = build_preprocessor(training, features)
        prep.fit(clean_features(training, features))
        return {
            kind: FinishCandidate(
                kind,
                features,
                prep,
                ToyScore(1 if kind == "lightgbm_ranker" else -1),
                kind == "lightgbm_ranker",
            )
            for kind in kinds
        }

    monkeypatch.setattr(recipe, "fit_finish_candidates", fit_models)
    monkeypatch.setattr(recipe, "build_dnf_classifier", lambda **kwargs: ToyDnf())
    first = study.run_deployment_study(
        stage, production, output, simulations=100, warmup_races=5, workers=1
    )
    assert len(calls) == 9 * 2 * 2
    assert not production.root.exists()
    audit = json.loads((output / "audit.json").read_text())
    assert audit["passed"] and audit["serialization_replay_max_error"] == 0
    results = pd.read_parquet(output / "predictions.parquet")
    saved = results.loc[
        results["mode"].eq("pre_quali")
        & results.candidate.eq(study.BASELINE)
        & results.season.eq(2026)
    ]
    replay = study.replay_deployment_candidate(stage, output, "pre_quali", study.BASELINE)
    assert study._prediction_error(saved, replay) == 0
    sensitivity = pd.read_csv(output / "simulation_sensitivity.csv")
    original_seed = sensitivity.loc[
        sensitivity["mode"].eq("pre_quali")
        & sensitivity.candidate.eq(study.BASELINE)
        & sensitivity.simulations.eq(100)
        & sensitivity.seed.eq(44)
    ].iloc[0]
    assert original_seed.position_log_loss == pytest.approx(
        study._metrics(saved)["position_log_loss"]
    )
    exporter = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/export_deployment_checkpoint.py")
    )
    exported = exporter["export_checkpoint"](output, stage, production)
    assert exported["passed"] and exported["stage2_matrices_unchanged"]
    assert exported["selected_cli_replay_max_error"] == 0
    monkeypatch.setattr(
        recipe, "fit_finish_candidates", lambda *a, **k: pytest.fail("resume retrained")
    )
    second = study.run_deployment_study(
        stage, production, output, simulations=100, warmup_races=5, workers=1
    )
    assert first["protocol"]["run_id"] == second["protocol"]["run_id"]
    assert first["metrics"] == second["metrics"]
    atomic_json_write(stage.features / "manifest.json", feature_manifest | {"revision": 2})
    monkeypatch.setattr(recipe, "fit_finish_candidates", fit_models)
    revised = study.run_deployment_study(
        stage, production, output, simulations=100, warmup_races=5, workers=1
    )
    assert revised["protocol"]["run_id"] != first["protocol"]["run_id"]


def test_loss_weights_are_passed_to_every_full_size_member_after_sorting(monkeypatch):
    from f1predict.models import candidates

    calls = []

    class CaptureModel:
        def __init__(self, **kwargs):
            self.options = kwargs

        def fit(self, matrix, target, **kwargs):
            calls.append((self.options, kwargs))

    for name in ("RandomForestRegressor", "LGBMRegressor", "LGBMRanker"):
        monkeypatch.setattr(candidates, name, CaptureModel)
    frame = toy_data().iloc[:8].sample(frac=1, random_state=12)
    weights = np.arange(1, 9, dtype=float)
    _, expected = _weighted_order(frame, weights)
    candidates.fit_finish_candidates(
        frame,
        ["driver_form"],
        tuple(recipe.ENSEMBLE_WEIGHTS["pre_quali"]),
        profile="full",
        sample_weight=weights,
        n_jobs=1,
    )
    assert {options["n_estimators"] for options, _ in calls} == {350, 450, 500}
    for options, fit_kwargs in calls:
        assert options["n_jobs"] == 1
        np.testing.assert_array_equal(fit_kwargs["sample_weight"], expected)
    assert calls[0][1]["group"] == [4, 4]


def test_real_bundle_cannot_predict_its_own_training_race(monkeypatch):
    data = toy_data()
    bundle = recipe.DeploymentBundle(
        "pre_quali",
        recipe.DeploymentConfig(),
        {},
        [],
        None,
        None,
        None,
        {},
        {"training_end": "2026-03-01"},
    )
    with pytest.raises(ValueError, match="strictly after"):
        bundle.raw_scores(data.loc[data.season.eq(2026)])
    with pytest.raises(ValueError, match="Outcome labels"):
        recipe.fit_deployment_bundle(
            data,
            ["target_win"],
            ["driver_form"],
            "pre_quali",
            recipe.DeploymentConfig(),
            pd.Timestamp("2024-01-01"),
            [],
        )
    with pytest.raises(ValueError, match="post-quali"):
        recipe.fit_deployment_bundle(
            data,
            ["post_quali_position"],
            ["driver_form"],
            "pre_quali",
            recipe.DeploymentConfig(),
            pd.Timestamp("2024-01-01"),
            [],
        )


def test_progress_reader_counts_only_completed_markers(tmp_path):
    progress_script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/deployment_progress.py")
    )
    output, stage = tmp_path / "outputs", tmp_path / "study/data"
    atomic_json_write(
        output / "protocol.json",
        {
            "run_id": "test",
            "warmup_races": 5,
            "evaluation_races": 4,
            "configurations": [recipe.DeploymentConfig().to_dict()],
            "modes": list(study.MODES),
        },
    )
    directory = stage.parent / "deployment_study/test/pre_quali/base_2018_uniform"
    atomic_json_write(directory / "2022_01/complete.json", {})
    atomic_json_write(directory / "2022_02/pending.json", {})
    result = progress_script["checkpoint_progress"](output, stage)
    assert result["completed_fits"] == 1
    assert result["total_fits"] == 18
    assert not result["final_report_present"]


def test_replay_driver_query_accepts_asof_bundle_without_driver_code(monkeypatch):
    from typer.testing import CliRunner

    from f1predict.cli import app

    frame = toy_data().loc[lambda d: d.season.eq(2026)]
    scores = {kind: frame.driver_form.to_numpy() for kind in recipe.MEMBERS}
    forecast = recipe.forecast_scores(
        frame,
        scores,
        np.full(4, 0.1),
        {kind: 0.85 for kind in recipe.MEMBERS},
        "pre_quali",
        simulations=100,
        seed=44,
    )
    assert "driver_code" not in forecast
    monkeypatch.setattr(study, "replay_deployment_candidate", lambda *a, **k: forecast)
    result = CliRunner().invoke(
        app, ["model", "replay-deployment", "--mode", "pre_quali", "--driver", "Driver 0"]
    )
    assert result.exit_code == 0, result.output
    assert "Driver 0" in result.output
    assert "Expected position" in result.output
    assert "Probability" in result.output
