"""Joint race simulations that propagate, rather than plug in, grid uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1predict.models.qualifying import standardize_scores


def position_matrix(
    strength_scenarios: np.ndarray,
    dnf_probability: np.ndarray,
    *,
    simulations: int = 4096,
    temperature: float = 0.75,
    seed: int = 44,
    smoothing: float = 0.001,
) -> np.ndarray:
    """Uniform mixture over coherent grid scenarios, with coherent uniform smoothing."""
    strengths = np.asarray(strength_scenarios, dtype=float)
    if strengths.ndim == 1:
        strengths = strengths[None, :]
    size = strengths.shape[1]
    dnf = np.asarray(dnf_probability, dtype=float)
    if (
        size < 2
        or simulations < 100
        or dnf.shape != (size,)
        or temperature <= 0
        or not 0 <= smoothing < 1
    ):
        raise ValueError("Invalid joint race simulation inputs")
    rng = np.random.default_rng(seed)
    standardized = standardize_scores(strengths)
    # Balanced scenario allocation avoids extra multinomial integration noise.
    sampled = standardized[np.arange(simulations) % len(strengths)]
    failures = rng.random((simulations, size)) < np.clip(dnf, 0.001, 0.999)
    all_failed = failures.all(axis=1)
    failures[np.flatnonzero(all_failed), np.argmax(sampled[all_failed], axis=1)] = False
    scores = sampled + rng.gumbel(scale=temperature, size=sampled.shape)
    scores -= failures * (4.0 + rng.uniform(size=sampled.shape))
    order = np.argsort(-scores, axis=1, kind="stable")
    positions = np.empty_like(order)
    positions[np.arange(simulations)[:, None], order] = np.arange(1, size + 1)
    probabilities = np.column_stack([(positions == p).mean(axis=0) for p in range(1, size + 1)])
    probabilities = (1 - smoothing) * probabilities + smoothing / size
    if not (
        np.allclose(probabilities.sum(axis=0), 1) and np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise AssertionError("Joint position matrix is not doubly stochastic")
    return probabilities


def prediction_frame(
    identifiers: pd.DataFrame,
    matrix: np.ndarray,
    dnf_probability: np.ndarray,
) -> pd.DataFrame:
    size = len(identifiers)
    frame = identifiers.reset_index(drop=True).copy()
    frame["expected_position"] = matrix @ np.arange(1, size + 1)
    frame["predicted_position"] = frame["expected_position"].rank(method="first").astype(int)
    frame["p_win"] = matrix[:, 0]
    frame["p_podium"] = matrix[:, : min(3, size)].sum(axis=1)
    frame["p_points"] = matrix[:, : min(10, size)].sum(axis=1)
    frame["p_dnf"] = dnf_probability
    for index in range(size):
        frame[f"p_position_{index + 1}"] = matrix[:, index]
    return frame.sort_values("predicted_position", ignore_index=True)
