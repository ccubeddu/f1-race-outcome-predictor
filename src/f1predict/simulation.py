from __future__ import annotations

import numpy as np
import pandas as pd


def _race_standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def simulate_race(
    drivers: pd.DataFrame,
    strength: np.ndarray,
    dnf_probability: np.ndarray,
    *,
    simulations: int = 20_000,
    temperature: float = 0.75,
    dnf_penalty: float = 4.0,
    seed: int = 44,
) -> pd.DataFrame:
    """Produce a coherent joint finishing distribution for one full grid."""
    if simulations < 100:
        raise ValueError("simulations must be at least 100")
    size = len(drivers)
    if size < 2:
        raise ValueError("a race simulation requires at least two drivers")
    strength = np.asarray(strength, dtype=float)
    dnf_probability = np.asarray(dnf_probability, dtype=float)
    if strength.shape != (size,) or dnf_probability.shape != (size,):
        raise ValueError("strength and DNF probabilities must match the grid size")

    rng = np.random.default_rng(seed)
    standardized = _race_standardize(strength)
    sampled_dnf = rng.random((simulations, size)) < np.clip(dnf_probability, 0.001, 0.999)
    all_dnf = sampled_dnf.all(axis=1)
    if all_dnf.any():
        sampled_dnf[all_dnf, int(np.argmax(standardized))] = False

    performance = standardized + rng.gumbel(
        loc=0.0,
        scale=max(float(temperature), 0.05),
        size=(simulations, size),
    )
    failure_order = rng.uniform(0.0, 1.0, size=(simulations, size))
    performance -= sampled_dnf * (dnf_penalty + failure_order)
    order = np.argsort(-performance, axis=1)
    positions = np.empty_like(order)
    positions[np.arange(simulations)[:, None], order] = np.arange(1, size + 1)

    probabilities = np.column_stack(
        [(positions == position).mean(axis=0) for position in range(1, size + 1)]
    )
    result = drivers.reset_index(drop=True).copy()
    result["expected_position"] = (probabilities * np.arange(1, size + 1)).sum(axis=1)
    result["p_win"] = probabilities[:, 0]
    result["p_podium"] = probabilities[:, : min(3, size)].sum(axis=1)
    result["p_points"] = probabilities[:, : min(10, size)].sum(axis=1)
    result["p_dnf"] = np.clip(dnf_probability, 0.001, 0.999)
    for index in range(size):
        result[f"p_position_{index + 1}"] = probabilities[:, index]
    result["predicted_position"] = (
        result["expected_position"].rank(method="first", ascending=True).astype(int)
    )

    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise AssertionError("each driver's position probabilities must sum to one")
    if not np.allclose(probabilities.sum(axis=0), 1.0):
        raise AssertionError("each finishing position must have exactly one driver")
    if not np.isclose(result["p_win"].sum(), 1.0):
        raise AssertionError("grid win probabilities must sum to one")
    return result.sort_values("predicted_position", ignore_index=True)
