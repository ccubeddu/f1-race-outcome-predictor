from __future__ import annotations

import numpy as np
import pandas as pd

from f1predict.models.ensemble import apply_pole_win_boost, blend_simulations
from f1predict.simulation import simulate_race


def test_simulation_returns_coherent_grid_probabilities() -> None:
    drivers = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "driver_name": ["A", "B", "C", "D"],
        }
    )
    result = simulate_race(
        drivers,
        strength=np.array([2.0, 1.0, 0.0, -1.0]),
        dnf_probability=np.array([0.05, 0.1, 0.15, 0.2]),
        simulations=5_000,
        seed=7,
    )
    position_columns = [f"p_position_{position}" for position in range(1, 5)]
    assert np.allclose(result[position_columns].sum(axis=1), 1.0)
    assert np.allclose(result[position_columns].sum(axis=0), 1.0)
    assert np.isclose(result["p_win"].sum(), 1.0)
    assert result["predicted_position"].tolist() == [1, 2, 3, 4]


def test_simulation_supports_dynamic_grid_size() -> None:
    drivers = pd.DataFrame({"driver_id": [f"d{index}" for index in range(22)]})
    result = simulate_race(
        drivers,
        strength=np.linspace(2, -2, 22),
        dnf_probability=np.full(22, 0.1),
        simulations=500,
    )
    assert "p_position_22" in result
    assert len(result) == 22


def test_anchored_ensemble_is_coherent_and_preserves_event_probabilities() -> None:
    drivers = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "driver_name": ["A", "B", "C", "D"],
        }
    )
    anchor = simulate_race(
        drivers,
        strength=np.array([2.0, 1.0, 0.0, -1.0]),
        dnf_probability=np.full(4, 0.1),
        simulations=5_000,
        seed=7,
    )
    challenger = simulate_race(
        drivers,
        strength=np.array([0.0, 2.0, -1.0, 1.0]),
        dnf_probability=np.full(4, 0.1),
        simulations=5_000,
        seed=8,
    )
    result = blend_simulations(
        {"lightgbm_ranker": anchor, "random_forest": challenger},
        {"lightgbm_ranker": 0.6, "random_forest": 0.4},
    )
    positions = [f"p_position_{position}" for position in range(1, 5)]
    aligned = result.set_index("driver_id").loc[anchor.set_index("driver_id").index]

    assert np.allclose(aligned[positions].sum(axis=1), 1.0)
    assert np.allclose(aligned[positions].sum(axis=0), 1.0)
    assert np.allclose(aligned["p_win"], anchor.set_index("driver_id")["p_win"])
    assert np.allclose(aligned["p_podium"], anchor.set_index("driver_id")["p_podium"])


def test_pole_boost_preserves_joint_probability_constraints() -> None:
    drivers = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "post_quali_position": [2, 1, 3, 4],
        }
    )
    original = simulate_race(
        drivers,
        strength=np.array([2.0, 1.0, 0.0, -1.0]),
        dnf_probability=np.full(4, 0.1),
        simulations=10_000,
        seed=9,
    )
    boosted = apply_pole_win_boost(original, factor=1.25).set_index("driver_id")
    baseline = original.set_index("driver_id")
    positions = [f"p_position_{position}" for position in range(1, 5)]

    assert boosted.loc["b", "p_win"] > baseline.loc["b", "p_win"]
    assert np.allclose(boosted["p_podium"], baseline["p_podium"])
    assert np.allclose(boosted[positions].sum(axis=1), 1.0)
    assert np.allclose(boosted[positions].sum(axis=0), 1.0)
