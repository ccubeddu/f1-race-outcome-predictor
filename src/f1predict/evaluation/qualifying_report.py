"""Human-readable checkpoint and calibration figures for the grid experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1predict.data.io import atomic_text_write

NAMES = {
    "direct_ensemble": "Direct ensemble (baseline)",
    "direct_ranker": "Direct LambdaRank anchor",
    "basic_practice_ranker": "Basic practice only (LambdaRank)",
    "grid_marginal_ensemble": "Qualifying-grid mixture ensemble",
    "grid_marginal_ranker": "Qualifying-grid mixture anchor",
}


def write_qualifying_report(
    report: dict[str, Any],
    calibration: pd.DataFrame,
    qualifying: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ("direct_ensemble", "basic_practice_ranker", "grid_marginal_ensemble")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, outcome in zip(axes, ("win", "podium", "points"), strict=True):
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        for candidate in variants:
            group = calibration.loc[
                calibration["candidate"].eq(candidate) & calibration["outcome"].eq(outcome)
            ]
            axis.plot(
                group["predicted"], group["observed"], "o-", markersize=4, label=NAMES[candidate]
            )
        axis.set(
            title=outcome.title(),
            xlabel="Predicted probability",
            ylabel="Observed frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Earlier-only calibration · {report['parameters']['profile']} model profile")
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)

    pole_rows = []
    for _, group in qualifying.groupby(["season", "round"]):
        actual = group["label_quali_position"]
        if actual.isna().any() or not actual.gt(0).all():
            continue
        pole = actual.eq(actual.min()).astype(int)
        pole_rows.append(pd.DataFrame({"probability": group["p_pole"], "actual": pole}))
    pole = pd.concat(pole_rows, ignore_index=True)
    pole["bin"] = pd.cut(pole["probability"], np.linspace(0, 1, 11), include_lowest=True)
    reliability = pole.groupby("bin", observed=True).agg(
        predicted=("probability", "mean"), observed=("actual", "mean"), count=("actual", "size")
    )
    atomic_text_write(
        output / "qualifying_calibration.csv", reliability.reset_index().to_csv(index=False)
    )
    fig, axis = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.plot(reliability["predicted"], reliability["observed"], "o-")
    axis.set(
        title="Qualifying: P(pole)",
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    fig.savefig(output / "qualifying_calibration.png", dpi=160)
    plt.close(fig)

    params = report["parameters"]
    lines = [
        "# Stage 3 — Qualifying uncertainty experiment",
        "",
        f"Retrospective walk-forward comparison, {params['start_season']}–{params['end_season']}. "
        f"Profile: **{params['profile']}**. "
        f"Each forecast integrates {params['grid_scenarios']} qualifying scenarios and "
        f"{params['simulations']:,} race simulations.",
        "",
        "Production data and models are unchanged. No candidate is automatically promoted.",
        "",
        "## Race predictions",
        "",
        "| Candidate | Races | Spearman ↑ | Top 3 ↑ | Top 10 ↑ | "
        "Pos. MAE ↓ | Pos. log loss ↓ | Winner log loss ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in NAMES:
        m = report["candidates"][candidate]
        lines.append(
            f"| {NAMES[candidate]} | {m['races']} | {m['spearman']:.4f} | "
            f"{m['top3_accuracy']:.1%} | {m['top10_accuracy']:.1%} | "
            f"{m['position_mae']:.3f} | {m['position_log_loss']:.4f} | {m['winner_log_loss']:.4f} |"
        )
    lines += [
        "",
        "Top-three/top-ten accuracy means set overlap, not exact ordered finishes. "
        "Winner log loss is race-level categorical loss, not per-driver binary log loss. "
        "The JSON includes binary win/podium/points/DNF log losses and Brier scores, "
        "categorical position Brier scores, and per-season results.",
        "",
        "## Paired uncertainty",
        "",
        "Grid-mixture ensemble minus direct ensemble; approximate paired four-race moving-block "
        "bootstrap, 2,000 replicates. Negative loss/MAE deltas are better.",
        "",
        "| Metric | Mean delta | 95% interval |",
        "|---|---:|---:|",
    ]
    for metric, values in report["paired_comparisons"]["grid_marginal_ensemble"].items():
        lo, hi = values["ci95"]
        lines.append(
            f"| {metric} | {values['candidate_minus_direct']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |"
        )
    q = report["qualifying"]
    lines += [
        "",
        "## Qualifying model",
        "",
        f"Scored {q['races']} races with complete archived qualifying labels: "
        f"Spearman **{q['spearman']:.4f}**, MAE **{q['position_mae']:.3f}**, "
        f"pole accuracy **{q['pole_accuracy']:.1%}**, pole log loss **{q['pole_log_loss']:.4f}**, "
        f"position log loss **{q['position_log_loss']:.4f}**.",
        "",
        "The qualifying ranker sees pre-Q inputs only. Conditional race models learn "
        "P(race | pre-Q features, historical qualifying order) on earlier races. "
        "At inference they receive sampled full-grid permutations, never the actual test Q. "
        "Monte Carlo averaging propagates grid uncertainty instead of plugging in a mean grid.",
        "",
        "No generated qualifying feature is used during race-model training, so no "
        "in-sample stacked prediction can leak in. All qualifying forecasts used for "
        "temperature calibration are chronological out-of-fold forecasts.",
        "",
        "## Protocol",
        "",
        *[f"- {s}" for s in report["protocol"]],
        "",
        "Temperature choices and DNF calibration update after each forecast is frozen. "
        "Shared chronological folds and DNF probabilities isolate the grid/ranking changes. "
        "The basic-practice ablation should be compared with the direct **ranker**, "
        "not the ensemble, to isolate detailed practice features.",
        "",
        "## Feature importance",
        "",
        "Final held-out fold models only; training ends before that race. Global gain importance "
        "is descriptive, not causal and not per-driver SHAP.",
        "",
    ]
    for name in ("qualifying", "direct", "conditional"):
        importance = pd.read_csv(output / f"importance_{name}.csv").head(10)
        lines += [f"### {name.title()}", "", "| Feature | Gain share |", "|---|---:|"]
        lines += [f"| {row.feature} | {row.importance:.1%} |" for row in importance.itertuples()]
        lines.append("")
    delta = report["paired_comparisons"]["grid_marginal_ensemble"]
    improved = delta["position_log_loss"]["ci95"][1] < 0 and delta["winner_log_loss"]["ci95"][1] < 0
    lines += [
        "## Decision",
        "",
        (
            "The grid mixture improved both primary losses "
            "with intervals below zero in this experiment. "
            if improved
            else "This experiment does not establish a robust improvement in both primary losses. "
        )
        + "Keep production frozen. Full deployment-profile checks and prospective forecasts "
        "are still required before any promotion.",
        "",
        "## Limitations",
        "",
        *[f"- {s}" for s in report["limitations"]],
        "",
        "## Artifacts",
        "",
        "- `predictions.parquet`: all held-out full-grid race distributions and labels.",
        "- `qualifying_predictions.parquet`: full qualifying distributions and labels.",
        "- `cutoffs.csv`: every training cutoff, including calibration warmup races.",
        "- `race_scores.csv`, `qualifying_scores.csv`, `metrics.json`: reproducible scoring.",
        "- `calibration.png`, `qualifying_calibration.png`: reliability plots.",
        "- `importance_*.csv`: all feature-gain shares.",
        "",
        "Method references: [LightGBM LambdaRank API]"
        "(https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) "
        "and [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).",
        "",
    ]
    atomic_text_write(output / "report.md", "\n".join(lines))
