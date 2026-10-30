"""Human-readable, non-promotional reporting of the fixed deployment study."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from f1predict.data.io import atomic_text_write
from f1predict.evaluation.deployment_study import BASELINE, MODES


def write_deployment_report(report: dict, calibration: pd.DataFrame, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "f1predict-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "f1predict-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [
        "# Stage 5 — matched full-size deployment experiments",
        "",
        "Research checkpoint only. Production data, feature matrices and models stay frozen. "
        "These are full-size **refits on staged v2 features**, not the old deployed v1 artifact.",
        "",
        "## Frozen protocol",
        "",
        "- Train before every race on the same archived entrants at both pre/post-quali cutoffs.",
        "- Compare 2018+, 2014+ and 2010+ windows, uniform or fixed 730.5-day half-life weights.",
        "- Test ratings + circuit with 2010+ uniform/decay models. DNF inputs stay at base schema.",
        "- Use 500 ranker, 350 forest, 450 regression and 400 DNF trees; one thread per fit.",
        "- Normalize loss weights to mean one; preprocessing is unweighted on training rows.",
        "- Share one fit/calibrate/predict recipe between backtests and saved bundles. "
        "DNF Platt calibration and simulation-based temperatures use 30 earlier OOF races.",
        f"- Start with {report['protocol']['warmup_races']} earlier warmup races; use "
        f"{report['protocol']['simulations']:,} joint draws with coherent uniform smoothing.",
        "- Select on 2023–2025 position loss, subject to fixed regression tolerances: "
        "Spearman −0.005; winner loss +0.03; top-3/top-10 overlap −2 pp; DNF Brier +0.001.",
        "- 2026 is a **reused retrospective check**, not an untouched holdout or selection set. "
        "Bootstrap intervals are approximate, without repeated-selection correction.",
        "- Post-quali keeps the inherited 1.25 pole adjustment and ensemble weights fixed. "
        "These settings have been tuned in earlier work, including reused seasons.",
        "",
        "The baseline uses 2018+ staged features. Thus history comparisons isolate training-row "
        "windows, not a completely new feature recipe. Stage 3 filtered training to observed "
        "qualifying labels for its conditional model; this direct study trains on all earlier "
        "race entrants. Its numbers are not an exact continuation of that experiment.",
        "",
    ]
    for mode in MODES:
        decision = report["decisions"][mode]
        selected = decision["candidate"]
        lines += [
            f"## {mode}",
            "",
            f"Development-selected candidate: **{selected}**. "
            f"Recommendation: **{decision['recommendation']}**. No automatic promotion.",
            "",
        ]
        for split, label in (
            ("development", "2023–2025 selection window"),
            ("retrospective", "2026 reused retrospective check"),
            ("all", "All 2023–2026 races — selection-biased summary"),
        ):
            lines += [
                f"### {label}",
                "",
                "| Candidate | Races | Spearman ↑ | Top 3 ↑ | Top 10 ↑ | MAE ↓ | "
                "Position LL ↓ | Winner LL ↓ | DNF Brier ↓ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for name, values in report["metrics"][mode][split].items():
                lines.append(
                    f"| {name} | {values['races']} | {values['spearman']:.4f} | "
                    f"{values['top3_accuracy']:.1%} | {values['top10_accuracy']:.1%} | "
                    f"{values['position_mae']:.3f} | {values['position_log_loss']:.4f} | "
                    f"{values['winner_log_loss']:.4f} | "
                    f"{values['probabilities']['dnf']['brier']:.5f} |"
                )
            lines.append("")
        lines += ["### Selected candidate checks", "", "| Check | Passed |", "|---|---|"]
        lines += [
            f"| {name} | {value} |" for name, value in decision["retrospective_checks"].items()
        ]
        lines += [
            "",
            "Selected minus baseline, approximate paired four-race block-bootstrap 95% intervals:",
            "",
        ]
        if selected != BASELINE:
            for metric, values in decision["paired_comparisons"]["retrospective"][selected].items():
                lo, hi = values["ci95"]
                lines.append(
                    f"- {metric}: {values['candidate_minus_direct']:+.4f} [{lo:+.4f}, {hi:+.4f}]."
                )
        else:
            lines.append(
                "Baseline retained; selected-versus-baseline differences are zero by construction."
            )
        lines.append("")
        if "simulation_sensitivity" in report:
            sensitivity = report["simulation_sensitivity"][mode]
            lo, hi = sensitivity["higher_draw_loss_delta_range"]
            lines += [
                f"Higher-draw position-loss difference across three seeds: [{lo:+.4f}, {hi:+.4f}]. "
                "Only prediction draws vary, not model fits or calibration-seed selection.",
                "",
            ]
        names = list(dict.fromkeys([BASELINE, selected, "ratings_2010_uniform"]))
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for axis, outcome in zip(axes.flat, ("win", "podium", "points", "dnf"), strict=True):
            axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            for name in names:
                group = calibration.loc[
                    calibration["mode"].eq(mode)
                    & calibration.candidate.eq(name)
                    & calibration.outcome.eq(outcome)
                ]
                axis.plot(group.predicted, group.observed, "o-", markersize=4, label=name)
            axis.set(
                title=outcome.upper(),
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Predicted probability",
                ylabel="Observed frequency",
            )
            axis.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Full-size {mode} · all reused test races · sparse tail bins need caution")
        fig.savefig(output / f"calibration_{mode}.png", dpi=160)
        plt.close(fig)
        importance = pd.read_csv(output / f"importance_{mode}_{selected}.csv").head(12)
        lines += [
            "### Selected-model feature importance",
            "",
            "Final as-of fold LightGBM gain; global and predictive, not causal or per-driver SHAP.",
            "",
            "| Feature | Gain share |",
            "|---|---:|",
        ]
        lines += [f"| {row.feature} | {row.importance:.1%} |" for row in importance.itertuples()]
        lines.append("")
    lines += [
        "## Interpretation and limitations",
        "",
        "A broader history may help stable driver effects while mixing incompatible regulation "
        "eras; recency weighting is tested rather than assumed to improve it. Rating states still "
        "summarize pre-event history from 2010 even when a model training-row window begins later "
        "(only the 2010+ models use ratings in this fixed experiment).",
        "",
        "Archived entrants are not a pre-quali entry-list forecast. Source publication/revision "
        "timestamps are not reconstructed. Practice eligibility and identity limits from stage 2, "
        "the excluded layout-date disagreements, coarse status labels and driver/car confounding "
        "remain. Top-three/top-ten accuracy is set overlap, not all-or-nothing race accuracy.",
        "",
        "Matrices are coherent and calibration is measured; temperature fitting "
        "does not guarantee perfect calibration of every podium/points/position probability. "
        "DNFs remain in finishing-order targets. A driver can rank first by expected "
        "position without giving that driver the highest win probability.",
        "",
        "The stage 3 grid mixture stays separate; neither its full-size superiority nor "
        "its combination with ratings is established. Sportsbooks, exchanges and markets "
        "are not combined, and no timing-eligible market advantage is claimed.",
        "",
        "No production files change even if a candidate passes a retrospective gate. A future "
        "promotion needs explicit approval, compatible live feature assembly, and genuinely frozen "
        "prospective forecasts. Saved bundles are as-of replay artifacts, not deployed models.",
        "",
        "Artifacts: `protocol.json`, `metrics.json`, `audit.json`, `predictions.parquet`, "
        "`cutoffs.csv`, "
        "`race_scores.csv`, `simulation_sensitivity.csv`, `calibration*.png`, `calibration.csv`, "
        "`feature_manifest.json`, `importance_*.csv`. Resumable models and raw scores stay under "
        "`work/history_expansion/deployment_study/<run_id>`.",
        "",
    ]
    atomic_text_write(output / "report.md", "\n".join(lines))
