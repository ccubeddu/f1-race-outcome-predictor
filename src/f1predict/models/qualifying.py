"""Qualifying ranking and coherent grid scenarios, without measured Q lap times."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from f1predict.models.candidates import FinishCandidate, fit_finish_candidate

CONDITIONAL_FEATURES = [
    "scenario_quali_position",
    "scenario_quali_percentile",
    "scenario_pole",
    "scenario_quali_vs_teammate",
]


def standardize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite qualifying/race strengths")
    scale = scores.std(axis=-1, keepdims=True)
    return (scores - scores.mean(axis=-1, keepdims=True)) / np.maximum(scale, 1e-9)


def sample_grid_positions(
    scores: np.ndarray,
    *,
    scenarios: int = 512,
    temperature: float = 0.75,
    seed: int = 44,
) -> np.ndarray:
    """Gumbel/Plackett–Luce sampling: every scenario is a full permutation of 1..N."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or scenarios < 1 or temperature <= 0:
        raise ValueError("Require >=2 drivers, positive scenarios and temperature")
    rng = np.random.default_rng(seed)
    order = np.argsort(
        -(
            standardize_scores(scores)
            + rng.gumbel(scale=temperature, size=(scenarios, len(scores)))
        ),
        axis=1,
        kind="stable",
    )
    positions = np.empty_like(order)
    positions[np.arange(scenarios)[:, None], order] = np.arange(1, len(scores) + 1)
    return positions


def plackett_luce_loss(scores: np.ndarray, positions: np.ndarray, temperature: float) -> float:
    """Mean sequential full-order NLL, evaluated on earlier held-out grids only."""
    scores = standardize_scores(scores) / temperature
    ordered = scores[np.argsort(positions, kind="stable")]
    return float(np.mean([logsumexp(ordered[i:]) - ordered[i] for i in range(len(ordered) - 1)]))


def choose_qualifying_temperature(
    history: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if len(history) < 5:
        return 0.75
    grid = (0.25, 0.4, 0.55, 0.75, 1.0, 1.3, 1.7, 2.2)
    return min(grid, key=lambda t: np.mean([plackett_luce_loss(s, p, t) for s, p in history]))


def qualifying_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """The target ranking is conditional on the known race-entrant universe.

    DNQ/non-entrant drivers are not added from post-event outcomes. Partial archived
    classifications train only their observed relative ranking, never invented labels.
    """
    data = frame.loc[frame["label_quali_position"].gt(0)].copy()
    groups = data.groupby(["season", "round"], observed=True)
    data["target_finish_position"] = groups["label_quali_position"].rank(method="first")
    data["field_size"] = groups["driver_id"].transform("size")
    return data.loc[data["field_size"].ge(2)].copy()


def fit_qualifying_model(
    historical: pd.DataFrame,
    features: list[str],
    *,
    profile: str = "full",
) -> FinishCandidate:
    if any(c.startswith(("post_", "scenario_", "label_", "target_")) for c in features):
        raise ValueError("Qualifying model inputs must be pre-qualifying features only")
    return fit_finish_candidate(
        qualifying_targets(historical), features, "lightgbm_ranker", profile=profile
    )


def conditional_inputs(pre_quali: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    """Build P(race | X, Q) inputs from ONLY X and the supplied scenario.

    At training time Q is observed in *earlier* races. At pre-Q inference Q is sampled;
    no current observed Q, Q1/2/3 times, final grid or grid penalties are consulted.
    No generated qualifying feature is used during race-model training, so there is
    no in-sample predicted-feature stacking. OOF grids are used for calibration/scoring.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(pre_quali),) or not np.isfinite(positions).all():
        raise ValueError("One finite qualifying position is required per driver")
    data = pre_quali.copy()
    data["scenario_quali_position"] = positions
    data["scenario_quali_percentile"] = (positions - 1) / np.maximum(
        data["field_size"].to_numpy(dtype=float) - 1, 1
    )
    data["scenario_pole"] = (positions == 1).astype("int8")
    groups = data.groupby(["season", "round", "constructor_id"], observed=True)
    count = groups["driver_id"].transform("size")
    teammate = (
        groups["scenario_quali_position"].transform("sum") - data["scenario_quali_position"]
    ) / (count - 1).replace(0, np.nan)
    data["scenario_quali_vs_teammate"] = positions - teammate
    return data


def scenario_strengths(
    model: FinishCandidate,
    pre_quali: pd.DataFrame,
    grids: np.ndarray,
) -> np.ndarray:
    """Score many grid scenarios in one model call; order remains scenario x driver."""
    if grids.ndim != 2 or grids.shape[1] != len(pre_quali):
        raise ValueError("Scenario grids must match the driver roster")
    frames = []
    for index, grid in enumerate(grids):
        frame = pre_quali.copy()
        # Synthetic grouping key keeps teammate calculations inside each scenario.
        frame["round"] = index
        frames.append(conditional_inputs(frame, grid))
    return model.predict_strength(pd.concat(frames, ignore_index=True)).reshape(grids.shape)


@dataclass
class GridForecast:
    positions: np.ndarray
    probabilities: np.ndarray

    @classmethod
    def from_positions(cls, positions: np.ndarray) -> GridForecast:
        size = positions.shape[1]
        probabilities = np.column_stack([(positions == p).mean(axis=0) for p in range(1, size + 1)])
        return cls(positions, probabilities)

    def to_frame(self, drivers: pd.DataFrame) -> pd.DataFrame:
        frame = drivers.reset_index(drop=True).copy()
        size = len(frame)
        frame["expected_quali_position"] = self.probabilities @ np.arange(1, size + 1)
        frame["predicted_quali_position"] = (
            frame["expected_quali_position"].rank(method="first").astype(int)
        )
        frame["p_pole"] = self.probabilities[:, 0]
        for index in range(size):
            frame[f"p_quali_position_{index + 1}"] = self.probabilities[:, index]
        return frame
