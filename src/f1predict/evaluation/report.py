from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "f1predict-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "f1predict-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from f1predict.config import DataPaths  # noqa: E402
from f1predict.data.io import atomic_json_write, atomic_text_write  # noqa: E402
from f1predict.models.training import load_model  # noqa: E402

DISPLAY_NAMES = {
    "pole_aware_ensemble": "Pole-aware ensemble",
    "anchored_ensemble": "Anchored ensemble",
    "random_forest": "Random Forest",
    "lightgbm_regression": "LightGBM regression",
    "lightgbm_ranker": "LightGBM LambdaRank",
}


def _load_mode(paths: DataPaths, mode: str) -> dict[str, Any]:
    path = paths.evaluation / mode / "metrics.json"
    if not path.exists():
        raise RuntimeError(f"Missing {mode} evaluation; run `f1predict evaluate run` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_plot(reports: dict[str, dict[str, Any]], target: Path) -> None:
    metrics = ["spearman", "top3_accuracy", "top10_accuracy", "position_mae"]
    titles = ["Spearman ↑", "Top-3 overlap ↑", "Top-10 overlap ↑", "Position MAE ↓"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#64748b", "#38bdf8", "#f43f5e"]
    for axis, metric, title in zip(axes.flat, metrics, titles, strict=True):
        width = 0.35
        candidates = [
            candidate
            for candidate in DISPLAY_NAMES
            if any(candidate in report["candidates"] for report in reports.values())
        ]
        x = np.arange(len(candidates))
        for mode_index, (mode, report) in enumerate(reports.items()):
            values = [
                report["candidates"].get(candidate, {}).get(metric, np.nan)
                for candidate in candidates
            ]
            axis.bar(
                x + (mode_index - 0.5) * width,
                values,
                width,
                label=mode.replace("_", " "),
                color=colors[mode_index + 1],
            )
        axis.set_title(title)
        axis.set_xticks(x, [DISPLAY_NAMES[value] for value in candidates], rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Walk-forward model comparison", fontsize=15, fontweight="bold")
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _calibration_plot(paths: DataPaths, mode: str, target: Path) -> None:
    table = pd.read_parquet(paths.evaluation / mode / "calibration.parquet")
    production_candidate = "pole_aware_ensemble" if mode == "post_quali" else "anchored_ensemble"
    table = table.loc[table["candidate"].eq(production_candidate)]
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for axis, outcome in zip(axes.flat, ("win", "podium", "points", "dnf"), strict=True):
        subset = table.loc[table["outcome"].eq(outcome)].sort_values("predicted")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
        sizes = np.clip(subset["count"].to_numpy() * 0.6, 20, 180)
        axis.scatter(
            subset["predicted"],
            subset["observed"],
            s=sizes,
            color="#f43f5e",
            alpha=0.85,
        )
        axis.plot(subset["predicted"], subset["observed"], color="#f43f5e", alpha=0.6)
        axis.set_title(outcome.upper())
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    figure.supxlabel("Predicted probability")
    figure.supylabel("Observed frequency")
    figure.suptitle(f"{mode.replace('_', ' ').title()} calibration — anchored ensemble")
    figure.tight_layout()
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _importance_plot(paths: DataPaths, target: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 7))
    for axis, mode in zip(axes, ("pre_quali", "post_quali"), strict=True):
        importance = load_model(paths, mode).feature_importance().head(15).sort_values("importance")
        axis.barh(importance["feature"], importance["importance"], color="#38bdf8")
        axis.set_title(mode.replace("_", " ").title())
        axis.set_xlabel("Normalized gain")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Final model feature importance", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown(reports: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Walk-forward evaluation",
        "",
        "Every test race was predicted by models trained only on earlier races. Probability "
        "calibration used only earlier out-of-fold predictions.",
        "",
        "| Mode | Model | Spearman | Top 3 | Top 10 | Position MAE | "
        "Position log loss | DNF Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, report in reports.items():
        for candidate, metrics in report["candidates"].items():
            lines.append(
                f"| {mode} | {DISPLAY_NAMES[candidate]} | {metrics['spearman']:.3f} | "
                f"{metrics['top3_accuracy']:.1%} | {metrics['top10_accuracy']:.1%} | "
                f"{metrics['position_mae']:.2f} | {metrics['position_log_loss']:.3f} | "
                f"{metrics['probabilities']['dnf']['brier']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Top-3 and top-10 accuracy are set-overlap recall within each race. Position log loss "
            "scores the probability assigned to the exact observed finishing position.",
            "",
            "## Production ensemble probability quality",
            "",
            "| Mode | Outcome | Log loss | Brier | Null Brier | Brier skill |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for mode, report in reports.items():
        production_candidate = (
            "pole_aware_ensemble" if mode == "post_quali" else "anchored_ensemble"
        )
        probabilities = report["candidates"][production_candidate]["probabilities"]
        for outcome, metrics in probabilities.items():
            lines.append(
                f"| {mode} | {outcome} | {metrics['log_loss']:.3f} | "
                f"{metrics['brier']:.3f} | {metrics['null_brier']:.3f} | "
                f"{metrics['brier_skill']:.1%} |"
            )
    lines.extend(
        [
            "",
            "The anchored ensemble blends structurally different rankers for finishing order. "
            "Post-qualifying predictions then apply a holdout-validated pole adjustment to "
            "P(win) while preserving P(podium), P(points), and joint race coherence.",
            "",
            "DNF Brier skill is close to zero. These probabilities are base-rate calibrated but "
            "have weak discrimination; improving component-level reliability data is a priority.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(paths: DataPaths, modes: tuple[str, ...]) -> dict[str, Any]:
    reports = {mode: _load_mode(paths, mode) for mode in modes}
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    _comparison_plot(reports, paths.evaluation / "model_comparison.png")
    for mode in modes:
        _calibration_plot(paths, mode, paths.evaluation / f"calibration_{mode}.png")
    if set(modes) == {"pre_quali", "post_quali"}:
        _importance_plot(paths, paths.evaluation / "feature_importance.png")
    summary = {
        "modes": reports,
        "artifacts": {
            "comparison_plot": "model_comparison.png",
            "calibration_plots": [f"calibration_{mode}.png" for mode in modes],
            "feature_importance": "feature_importance.png" if len(modes) == 2 else None,
        },
    }
    atomic_json_write(paths.evaluation / "summary.json", summary)
    atomic_text_write(paths.evaluation / "report.md", _markdown(reports))
    return summary
