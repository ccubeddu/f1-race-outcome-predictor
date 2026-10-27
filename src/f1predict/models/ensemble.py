from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR = "lightgbm_ranker"
POLE_WIN_FACTOR = 1.25
ENSEMBLE_WEIGHTS: dict[str, dict[str, float]] = {
    "pre_quali": {
        "lightgbm_ranker": 0.50,
        "random_forest": 0.30,
        "lightgbm_regression": 0.20,
    },
    "post_quali": {
        "lightgbm_ranker": 0.60,
        "random_forest": 0.30,
        "lightgbm_regression": 0.10,
    },
}


def _position_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame if column.startswith("p_position_")),
        key=lambda value: int(value.rsplit("_", maxsplit=1)[-1]),
    )[: len(frame)]


def _rake_block(values: np.ndarray, row_targets: np.ndarray) -> np.ndarray:
    """Fit a probability block to driver row totals and unit position columns."""
    if values.shape[1] == 1:
        return row_targets[:, None]
    result = np.maximum(values, 1e-12)
    for _ in range(200):
        row_sums = result.sum(axis=1)
        result *= np.divide(
            row_targets,
            row_sums,
            out=np.zeros_like(row_targets),
            where=row_sums > 0,
        )[:, None]
        column_sums = result.sum(axis=0)
        result *= np.divide(
            np.ones(values.shape[1]),
            column_sums,
            out=np.ones(values.shape[1]),
            where=column_sums > 0,
        )[None, :]
    return result


def blend_simulations(
    simulations: dict[str, pd.DataFrame],
    weights: dict[str, float],
    *,
    anchor: str = ANCHOR,
) -> pd.DataFrame:
    """Blend one-race simulations while preserving calibrated event probabilities.

    The LambdaRank member anchors probability mass in four disjoint finishing-position
    blocks: win, P2-P3, P4-P10, and outside the points. Iterative proportional fitting
    incorporates the challenger models inside each block while retaining one driver per
    position and the anchor's P(win), P(podium), and P(points) for every driver.
    """
    if anchor not in simulations:
        raise ValueError(f"Missing ensemble anchor: {anchor}")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Ensemble weights must sum to one")
    missing = {candidate for candidate, weight in weights.items() if weight > 0} - set(simulations)
    if missing:
        raise ValueError(f"Missing ensemble simulations: {sorted(missing)}")

    reference = simulations[anchor].set_index("driver_id").copy()
    position_columns = _position_columns(reference)
    mixture = np.zeros((len(reference), len(position_columns)), dtype=float)
    for candidate, weight in weights.items():
        if weight <= 0:
            continue
        member = simulations[candidate].set_index("driver_id").reindex(reference.index)
        mixture += weight * member[position_columns].fillna(0).to_numpy(dtype=float)

    anchor_matrix = reference[position_columns].fillna(0).to_numpy(dtype=float)
    boundaries = [0, 1, min(3, len(reference)), min(10, len(reference)), len(reference)]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end <= start:
            continue
        row_targets = anchor_matrix[:, start:end].sum(axis=1)
        mixture[:, start:end] = _rake_block(mixture[:, start:end], row_targets)

    reference[position_columns] = mixture
    reference["expected_position"] = mixture @ np.arange(1, len(reference) + 1)
    reference["p_win"] = mixture[:, 0]
    reference["p_podium"] = mixture[:, : min(3, len(reference))].sum(axis=1)
    reference["p_points"] = mixture[:, : min(10, len(reference))].sum(axis=1)
    reference["predicted_position"] = (
        reference["expected_position"].rank(method="first").astype(int)
    )

    if not np.allclose(mixture.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("each driver's ensemble distribution must sum to one")
    if not np.allclose(mixture.sum(axis=0), 1.0, atol=1e-8):
        raise AssertionError("each ensemble finishing position must sum to one")
    return reference.reset_index().sort_values("predicted_position", ignore_index=True)


def _rescale_with_caps(values: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Normalize positive scores to one without exceeding per-driver probability caps."""
    result = np.zeros_like(values, dtype=float)
    remaining = np.ones(len(values), dtype=bool)
    remaining_mass = 1.0
    for _ in range(len(values)):
        scores = values[remaining]
        if scores.sum() <= 0:
            scores = np.ones_like(scores)
        proposal = scores / scores.sum() * remaining_mass
        active_caps = caps[remaining]
        over = proposal > active_caps
        indices = np.flatnonzero(remaining)
        if not over.any():
            result[indices] = proposal
            return result
        capped_indices = indices[over]
        result[capped_indices] = caps[capped_indices]
        remaining[capped_indices] = False
        remaining_mass = 1.0 - result.sum()
    return result


def apply_pole_win_boost(
    prediction: pd.DataFrame,
    *,
    factor: float = POLE_WIN_FACTOR,
    position_column: str = "post_quali_position",
) -> pd.DataFrame:
    """Increase pole's win odds while preserving podium/points and joint coherence."""
    if factor <= 0:
        raise ValueError("pole win factor must be positive")
    if position_column not in prediction:
        raise ValueError(f"Missing qualifying position column: {position_column}")
    result = prediction.copy().set_index("driver_id")
    pole = result[position_column].eq(1).to_numpy()
    if pole.sum() != 1:
        raise ValueError("Pole-aware calibration requires exactly one qualifying winner")
    position_columns = _position_columns(result)
    matrix = result[position_columns].fillna(0).to_numpy(dtype=float)
    scores = matrix[:, 0] * np.where(pole, factor, 1.0)
    win_probability = _rescale_with_caps(scores, result["p_podium"].to_numpy(dtype=float))
    matrix[:, 0] = win_probability
    podium_remainder = result["p_podium"].to_numpy(dtype=float) - win_probability
    matrix[:, 1:3] = _rake_block(matrix[:, 1:3], podium_remainder)

    result[position_columns] = matrix
    result["expected_position"] = matrix @ np.arange(1, len(result) + 1)
    result["p_win"] = matrix[:, 0]
    result["p_podium"] = matrix[:, : min(3, len(result))].sum(axis=1)
    result["p_points"] = matrix[:, : min(10, len(result))].sum(axis=1)
    result["predicted_position"] = result["expected_position"].rank(method="first").astype(int)
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("pole-aware driver distributions must sum to one")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
        raise AssertionError("pole-aware finishing positions must sum to one")
    return result.reset_index().sort_values("predicted_position", ignore_index=True)
