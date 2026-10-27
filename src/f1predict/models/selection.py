from __future__ import annotations

from typing import Any

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write


def production_selection() -> dict[str, Any]:
    return {
        "candidate": "pole_aware_ensemble",
        "model": "Pole-aware anchored three-model ensemble",
        "default_mode": "auto",
        "mode_policy": {
            "post_quali": "use when a qualifying classification is available",
            "pre_quali": "fallback before qualifying",
        },
        "reason": (
            "The three-model mixture improves top-three accuracy and position log loss; "
            "post-quali predictions add a holdout-validated pole adjustment while preserving "
            "coherent podium and points probabilities."
        ),
        "evaluation_window": "66 races, 2024-2026",
    }


def write_production_selection(paths: DataPaths) -> dict[str, Any]:
    selection = production_selection()
    atomic_json_write(paths.models / "selection.json", selection)
    return selection
