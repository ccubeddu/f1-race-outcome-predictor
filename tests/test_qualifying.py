from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write
from f1predict.evaluation.grid_simulation import position_matrix
from f1predict.evaluation.qualifying_study import _member_forecasts, _temperature, load_study_data
from f1predict.models.qualifying import (
    CONDITIONAL_FEATURES,
    GridForecast,
    choose_qualifying_temperature,
    conditional_inputs,
    fit_qualifying_model,
    plackett_luce_loss,
    qualifying_targets,
    sample_grid_positions,
    scenario_strengths,
)


def drivers(size=4):
    return pd.DataFrame(
        {
            "driver_id": [f"d{i}" for i in range(size)],
            "constructor_id": [str(i // 2) for i in range(size)],
            "season": [2024] * size,
            "round": [1] * size,
            "field_size": [size] * size,
            "race_date": [pd.Timestamp("2024-03-01")] * size,
            "race_name": ["Race"] * size,
            "driver_name": [f"D{i}" for i in range(size)],
            "constructor_name": [str(i // 2) for i in range(size)],
        }
    )


def test_grid_sampler_is_reproducible_permutation_and_dynamic_size() -> None:
    for size in (4, 20, 22, 24):
        scores = np.linspace(2, -2, size)
        positions = sample_grid_positions(scores, scenarios=2048, seed=7)
        assert np.all(np.sort(positions, axis=1) == np.arange(1, size + 1))
        assert np.array_equal(positions, sample_grid_positions(scores, scenarios=2048, seed=7))
        forecast = GridForecast.from_positions(positions).to_frame(drivers(size))
        matrix = forecast[[f"p_quali_position_{i}" for i in range(1, size + 1)]]
        assert np.allclose(matrix.sum(axis=0), 1)
        assert np.allclose(matrix.sum(axis=1), 1)
        assert np.isclose(forecast["p_pole"].sum(), 1)
        assert (
            forecast.loc[0, "expected_quali_position"]
            < forecast.loc[size - 1, "expected_quali_position"]
        )


def test_qualifying_temperature_uses_earlier_rank_labels() -> None:
    scores = np.array([3, 1, 0, -2])
    order = np.arange(1, 5)
    assert choose_qualifying_temperature([]) == 0.75
    assert plackett_luce_loss(scores, order, 0.25) < plackett_luce_loss(scores, order, 2)
    assert choose_qualifying_temperature([(scores, order)] * 5) == 0.25


def test_qualifying_training_rejects_post_features_and_ignores_race_targets() -> None:
    frame = drivers().assign(
        label_quali_position=[1, 3, 2, np.nan], target_finish_position=[4, 3, 2, 1]
    )
    data = qualifying_targets(frame)
    assert data["target_finish_position"].tolist() == [1, 3, 2]
    assert data["field_size"].tolist() == [3] * 3
    with pytest.raises(ValueError, match="pre-qualifying"):
        fit_qualifying_model(frame, ["post_quali_position"])


def test_conditional_inputs_use_only_supplied_grid_and_teammates() -> None:
    frame = drivers().assign(label_quali_position=[4, 3, 2, 1], q3_seconds=99, post_grid_position=1)
    inputs = conditional_inputs(frame, np.array([1, 4, 2, 3]))
    assert inputs["scenario_quali_vs_teammate"].tolist() == [-3, 3, -1, 1]
    changed = frame.assign(label_quali_position=[1, 2, 3, 4], q3_seconds=1, post_grid_position=99)
    pd.testing.assert_frame_equal(
        inputs[CONDITIONAL_FEATURES],
        conditional_inputs(changed, np.array([1, 4, 2, 3]))[CONDITIONAL_FEATURES],
    )

    class Model:
        def predict_strength(self, frame):
            return -frame["scenario_quali_position"].to_numpy()

    grids = np.array([[1, 4, 2, 3], [4, 1, 3, 2]])
    assert np.array_equal(scenario_strengths(Model(), frame, grids), -grids)


def test_joint_grid_race_integration_preserves_probability_constraints() -> None:
    grids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    matrix = position_matrix(-grids, np.full(4, 0.05), simulations=4000, seed=7)
    assert np.allclose(matrix.sum(axis=0), 1)
    assert np.allclose(matrix.sum(axis=1), 1)
    assert (matrix > 0).all()
    fixed = position_matrix(-grids[:1], np.full(4, 0.05), simulations=4000, seed=7)
    assert matrix[0, 0] < fixed[0, 0]
    assert matrix[3, 0] > fixed[3, 0]


def test_current_labels_cannot_change_current_temperature_or_prediction() -> None:
    frame = drivers().assign(
        target_finish_position=[1, 2, 3, 4],
        target_win=[1, 0, 0, 0],
        target_podium=[1, 1, 1, 0],
        target_points=1,
        target_dnf=0,
    )
    strengths = {"lightgbm_ranker": np.array([2, 1, 0, -1])}
    before, loss1 = _member_forecasts(
        frame, strengths, np.full(4, 0.1), {}, "test", simulations=1000, seed=1
    )
    flipped = frame.assign(target_win=[0, 0, 0, 1], target_finish_position=[4, 3, 2, 1])
    after, loss2 = _member_forecasts(
        flipped, strengths, np.full(4, 0.1), {}, "test", simulations=1000, seed=1
    )
    assert _temperature([]) == 0.85
    assert loss1 != loss2
    columns = ["p_win", "expected_position", "race_temperature"]
    pd.testing.assert_frame_equal(
        before["lightgbm_ranker"][columns], after["lightgbm_ranker"][columns]
    )


def test_study_labels_are_separate_from_feature_contract(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    atomic_json_write(
        paths.features / "manifest.json",
        {"schema_version": 2, "modes": {"pre_quali": {"features": ["field_size"]}}},
    )
    pre = drivers()
    post = pre.assign(post_quali_position=[1, 2, 3, 4])
    atomic_parquet_write(pre, paths.features / "pre_quali.parquet")
    atomic_parquet_write(post, paths.features / "post_quali.parquet")
    data, features = load_study_data(paths)
    assert features == ["field_size"]
    assert "label_quali_position" in data
    assert not any(c.startswith("post_") for c in data)
