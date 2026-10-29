"""Readable results and reliability plots for the rating-feature experiment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from f1predict.data.io import atomic_text_write
from f1predict.evaluation.rating_study import NAMES


def write_rating_report(
    report: dict, calibration: pd.DataFrame, manifest: dict, output: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "f1predict-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "f1predict-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    main = ("direct_ensemble", "ratings_ensemble", "reliability_ensemble")
    for axis, outcome in zip(axes.flat, ("win", "podium", "points", "dnf"), strict=True):
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        for candidate in main:
            group = calibration.loc[
                calibration.candidate.eq(candidate) & calibration.outcome.eq(outcome)
            ]
            axis.plot(group.predicted, group.observed, "o-", markersize=4, label=NAMES[candidate])
        axis.set(
            title=outcome.upper(),
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Predicted probability",
            ylabel="Observed frequency",
        )
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        f"Rating study · {report['parameters']['profile']} profile · sparse tail bins need caution"
    )
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Stage 4 — Driver, constructor and circuit ratings",
        "",
        f"Matched expanding-window pre-quali comparison, "
        f"{report['parameters']['start_season']}–{report['parameters']['end_season']}. "
        f"Tree profile: **{report['parameters']['profile']}**. Production artifacts unchanged; "
        "no candidate promoted.",
        "",
        "## Race metrics",
        "",
        "| Candidate | Races | Spearman ↑ | Top 3 ↑ | Top 10 ↑ | Pos. MAE ↓ | "
        "Position LL ↓ | Winner LL ↓ | DNF Brier ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        m = report["candidates"][name]
        lines.append(
            f"| {NAMES[name]} | {m['races']} | {m['spearman']:.4f} | "
            f"{m['top3_accuracy']:.1%} | {m['top10_accuracy']:.1%} | {m['position_mae']:.3f} | "
            f"{m['position_log_loss']:.4f} | {m['winner_log_loss']:.4f} | "
            f"{m['probabilities']['dnf']['brier']:.5f} |"
        )
    lines += [
        "",
        "Top-3/top-10 measure set overlap. Winner LL is race-level categorical loss. "
        "The JSON also includes binary log losses/Brier scores and position Brier scores.",
        "",
        "## Ablation protocol",
        "",
        "- Compare direct, global-ratings and circuit-ratings **rankers** "
        "to isolate feature additions.",
        "- Compare the direct and circuit-ratings **ensembles** "
        "for the matched model-stack comparison.",
        "- Those models share exactly the same base DNF probabilities.",
        "- The reliability variant reuses the circuit-rating finish models, changing only its DNF "
        "classifier inputs/calibration and downstream temperature calibration.",
        "- All models use the same training rows and held-out races, frozen settings, simulation "
        "seeds and earlier-only calibration. The direct baseline is refit on staged data, not "
        "the old frozen production artifact.",
        "",
        "## Paired uncertainty versus direct ensemble",
        "",
        "Candidate minus baseline; 95% approximate four-race moving-block bootstrap intervals. "
        "Lower losses/MAE are better; higher Spearman is better.",
        "",
        "| Candidate | Metric | Mean delta | 95% interval |",
        "|---|---|---:|---:|",
    ]
    for name in ("ratings_ensemble", "reliability_ensemble"):
        for metric, values in report["paired_comparisons"][name].items():
            lo, hi = values["ci95"]
            lines.append(
                f"| {NAMES[name]} | {metric} | {values['candidate_minus_direct']:+.4f} | "
                f"[{lo:+.4f}, {hi:+.4f}] |"
            )
    lines += [
        "",
        "## Regulation-era breakdown",
        "",
        "| Candidate | Era | Races | Spearman | Position LL | DNF Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in main:
        for era, values in report["by_era"][name].items():
            lines.append(
                f"| {NAMES[name]} | {era} | {values['races']} | {values['spearman']:.4f} | "
                f"{values['position_log_loss']:.4f} | "
                f"{values['probabilities']['dnf']['brier']:.5f} |"
            )
    lines += [
        "",
        "The JSON contains the full per-season results, not just era averages.",
        "",
        "## Feature design and leakage policy",
        "",
        f"Rating sidecar: {manifest['rows']:,} driver-race rows; "
        f"{len(manifest['families']['global'])} global, {len(manifest['families']['context'])} "
        f"circuit/layout, and {len(manifest['families']['reliability'])} reliability features.",
        "",
        "- All entrants are snapshotted before any event update. Qualifying ratings also "
        "update only after the race; current qualifying is not used by these pre-Q features.",
        "- A pairwise logistic update models driver + constructor strength. Driver updates "
        "include teammate comparisons. Race-rank updates exclude source-defined DNFs/DNS/DSQ "
        "and use classified finishers only; qualifying has its own channel.",
        "- Driver ratings persist through moves; constructor states stay with the operation. "
        "Time-bounded Enstone/Hinwil/Faenza/Silverstone mappings share discounted latent state, "
        "never official championship points. Other IDs stay separate; unrelated Lotus teams "
        "are not merged, and Cadillac starts without predecessor evidence.",
        "- Annual, regulation-era and rebrand discounts are fixed research priors, not fitted "
        "to these test seasons. All constants are in `feature_manifest.json`.",
        "- Circuit-layout and coarse geometry-bucket residuals shrink sparse evidence toward zero. "
        "They do not feed back into global rating updates, keeping the feature ablation clean.",
        "- Event-specific F1DB layout geometry is used, never today's circuit layout, actual "
        "race distance, future aggregate wins or championship-decider flags.",
        "- Mechanical-, incident- and other-labeled DNF histories are separate from pace. "
        "Dirichlet-smoothed component rates sum to total DNF rate. Generic 'Retired' stays "
        "unknown/other; incident labels do not establish driver fault.",
        "",
        "## Layout audit",
        "",
        f"Verified-date geometry coverage: {manifest['layout_coverage']:.2%}. "
        "Disagreeing dates are excluded from geometry attribution, using unique unknown "
        "layout keys so they cannot borrow another circuit's evidence.",
        "",
    ]
    for item in manifest["layout_exclusions"]:
        lines.append(
            f"- {item['season']} R{item['round']}: result date {str(item['race_date'])[:10]}, "
            f"layout archive date {item['layout_date']}; geometry unused."
        )
    lines += [
        "",
        "## Feature importance",
        "",
        "Global LightGBM gain from the final held-out fold only, trained before that race. "
        "These are predictive proxies, not causal driver/car decompositions or per-driver SHAP.",
        "",
    ]
    for name in ("global", "ratings"):
        importance = pd.read_csv(output / f"importance_{name}.csv").head(12)
        lines += [f"### {name.title()}", "", "| Feature | Gain share |", "|---|---:|"]
        lines += [f"| {r.feature} | {r.importance:.1%} |" for r in importance.itertuples()]
        lines.append("")
    lines += [
        "## Decision and limitations",
        "",
        "Keep production frozen at this checkpoint. A development-profile retrospective "
        "comparison is not proof of full deployment improvement. Driver and constructor "
        "effects are not uniquely identifiable from observational results; strategy, team orders, "
        "traffic and status-label noise remain confounders. Classification/status labels are "
        "coarse; these ratings are not fuel-corrected telemetry pace estimates.",
        "",
        "2026 remains reused retrospective validation. The test roster is the archived entrant "
        "set; source publication/revision timing is not reconstructed. Monte Carlo convergence "
        "across seeds and higher simulation counts is not established. Bootstrap intervals are "
        "approximate and not corrected for repeated model selection. Only pre-quali is evaluated "
        "here; full-profile pre/post-quali promotion checks belong to stage 5.",
        "",
        "Artifacts: `metrics.json`, `predictions.parquet`, `race_scores.csv`, `calibration.csv`, "
        "`calibration.png`, `rating_snapshots.parquet`, `feature_manifest.json`, `audit.json`, "
        "`cutoffs.csv` and `importance_*.csv`.",
        "",
        "Sources: [F1DB](https://github.com/f1db/f1db), "
        "[official team lineage history](https://www.formula1.com/en/latest/article/"
        "the-family-tree-f1-11-teams-and-how-they-came-to-be."
        "2QBA1PPMf0bC8mp2xxqZeq.2QBA1PPMf0bC8mp2xxqZeq), "
        "[regulation-era history](https://www.formula1.com/en/latest/article/"
        "the-key-regulation-changes-in-f1-history-and-the-teams-that-nailed-them."
        "2iq8c5E6S1HOffT5PBLl6i), "
        "and [2026 rules](https://www.formula1.com/en/page/2026-f1-regulations).",
        "",
    ]
    atomic_text_write(output / "report.md", "\n".join(lines))
