from __future__ import annotations

import numpy as np
import pandas as pd

from f1predict.evaluation.metrics import evaluate_predictions


def test_metrics_score_perfect_race_order() -> None:
    frame = pd.DataFrame(
        {
            "season": [2024] * 4,
            "round": [1] * 4,
            "driver_id": list("abcd"),
            "predicted_position": [1, 2, 3, 4],
            "target_finish_position": [1, 2, 3, 4],
            "target_win": [1, 0, 0, 0],
            "target_podium": [1, 1, 1, 0],
            "target_points": [1, 1, 1, 1],
            "target_dnf": [0, 0, 0, 1],
            "p_win": [0.9, 0.05, 0.03, 0.02],
            "p_podium": [0.99, 0.95, 0.9, 0.16],
            "p_points": [1.0, 1.0, 1.0, 1.0],
            "p_dnf": [0.01, 0.02, 0.03, 0.8],
            **{f"p_position_{position}": np.eye(4)[:, position - 1] for position in range(1, 5)},
        }
    )
    metrics = evaluate_predictions(frame)
    assert metrics["spearman"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["top10_accuracy"] == 1.0
    assert metrics["position_mae"] == 0.0
    assert metrics["position_log_loss"] == 0.0
