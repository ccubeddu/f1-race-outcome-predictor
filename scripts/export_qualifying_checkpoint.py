"""Audit a completed qualifying study and export an easy-to-read Austin example."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.history import _fingerprints
from f1predict.data.io import atomic_json_write, atomic_text_write


def export_checkpoint(study: Path, production: DataPaths, stage: DataPaths) -> dict:
    predictions = pd.read_parquet(study / "predictions.parquet")
    qualifying = pd.read_parquet(study / "qualifying_predictions.parquet")
    errors = dict(
        driver_total=0.0,
        position_total=0.0,
        win_total=0.0,
        podium_total=0.0,
        points_total=0.0,
        expected_position=0.0,
        summary=0.0,
    )
    reference_keys = None
    for _, candidate in predictions.groupby("candidate"):
        keys = set(candidate[["season", "round", "driver_id"]].itertuples(index=False, name=None))
        if reference_keys is not None and keys != reference_keys:
            raise AssertionError("Candidate test supports differ")
        reference_keys = keys
        for _, group in candidate.groupby(["season", "round"]):
            size = len(group)
            matrix = group[[f"p_position_{i}" for i in range(1, size + 1)]].to_numpy()
            if not np.isfinite(matrix).all() or matrix.min() < 0 or matrix.max() > 1:
                raise AssertionError("Invalid finishing probabilities")
            errors["driver_total"] = max(
                errors["driver_total"], float(abs(matrix.sum(axis=1) - 1).max())
            )
            errors["position_total"] = max(
                errors["position_total"], float(abs(matrix.sum(axis=0) - 1).max())
            )
            for key, column, total in (
                ("win_total", "p_win", 1),
                ("podium_total", "p_podium", min(3, size)),
                ("points_total", "p_points", min(10, size)),
            ):
                errors[key] = max(errors[key], abs(float(group[column].sum()) - total))
            expected = matrix @ np.arange(1, size + 1)
            errors["expected_position"] = max(
                errors["expected_position"], float(abs(expected - group.expected_position).max())
            )
            errors["summary"] = max(
                errors["summary"],
                float(abs(matrix[:, 0] - group.p_win).max()),
                float(abs(matrix[:, :3].sum(axis=1) - group.p_podium).max()),
                float(abs(matrix[:, : min(10, size)].sum(axis=1) - group.p_points).max()),
            )
    chronology = int(
        pd.to_datetime(predictions.training_end).ge(pd.to_datetime(predictions.race_date)).sum()
    )
    duplicates = int(predictions.duplicated(["candidate", "season", "round", "driver_id"]).sum())
    qerror = 0.0
    for _, group in qualifying.groupby(["season", "round"]):
        matrix = group[[f"p_quali_position_{i}" for i in range(1, len(group) + 1)]].to_numpy()
        qerror = max(
            qerror,
            float(abs(matrix.sum(axis=0) - 1).max()),
            float(abs(matrix.sum(axis=1) - 1).max()),
        )
    snapshot = json.loads((stage.root / "baseline_snapshot.json").read_text())
    unchanged = snapshot["production_fingerprints"] == _fingerprints(production)
    passed = (
        max(errors.values()) < 1e-8
        and qerror < 1e-8
        and chronology == duplicates == 0
        and unchanged
    )
    audit = dict(
        passed=passed,
        probability_errors=errors,
        qualifying_probability_error=qerror,
        chronology_violations=chronology,
        duplicate_keys=duplicates,
        production_unchanged=unchanged,
        races=int(predictions[["season", "round"]].drop_duplicates().shape[0]),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")
        },
    )
    atomic_json_write(study / "audit.json", audit)
    if not passed:
        raise AssertionError(f"Qualifying study audit failed: {audit}")

    report = json.loads((study / "metrics.json").read_text())
    direct = report["candidates"]["direct_ensemble"]
    mixture = report["candidates"]["grid_marginal_ensemble"]
    improvement = 100 * (1 - mixture["position_log_loss"] / direct["position_log_loss"])
    pole = pd.read_csv(study / "qualifying_calibration.csv")
    middle = pole.loc[pole["predicted"].between(0.1, 0.3)]
    count = int(middle["count"].sum())
    predicted_pole = float(np.average(middle["predicted"], weights=middle["count"]))
    observed_pole = float(np.average(middle["observed"], weights=middle["count"]))
    ranker = report["candidates"]["direct_ranker"]
    basic = report["candidates"]["basic_practice_ranker"]
    audit["monte_carlo_convergence_across_seeds_checked"] = False
    atomic_json_write(study / "audit.json", audit)
    assessment = [
        "# Stage 3 assessment",
        "",
        "The qualifying-grid model is useful for full finishing distributions, "
        "but is not an across-the-board upgrade. Keep production unchanged.",
        "",
        f"- Position log loss improves **{improvement:.1f}%**: "
        f"{direct['position_log_loss']:.4f} → {mixture['position_log_loss']:.4f}. "
        "The paired block-bootstrap interval favors the mixture, and its position log loss "
        "is lower in each of the four evaluated seasons.",
        f"- Winner log loss worsens: {direct['winner_log_loss']:.4f} → "
        f"{mixture['winner_log_loss']:.4f}; the paired uncertainty interval crosses zero.",
        f"- Rank correlation is essentially flat: {direct['spearman']:.4f} → "
        f"{mixture['spearman']:.4f}. Better exact-position probabilities "
        "do not imply a better ordering.",
        "",
        "## Calibration weakness",
        "",
        f"For {count} qualifying driver-race predictions in the 10–30% pole-probability "
        f"range, mean predicted probability is **{predicted_pole:.1%}**, versus "
        f"**{observed_pole:.1%}** observed pole frequency. "
        "The pole reliability curve therefore shows under-confidence in this range. "
        "Higher-probability bins are sparse and should not be overinterpreted.",
        "",
        "A sensible next research test is a separately calibrated first-place draw followed "
        "by a conditional ordering of the remaining drivers, with both calibrators fit only "
        "to earlier OOF forecasts. That could sharpen P(pole) while retaining uncertain lower "
        "grid positions. This has not been implemented or evaluated, and this same retrospective "
        "window would not become a fresh holdout for that test.",
        "",
        "## Practice ablation",
        "",
        "Compare the rankers, since the ensemble also changes within-position-block ordering. "
        f"Detailed versus basic practice position log loss is {ranker['position_log_loss']:.4f} "
        f"versus {basic['position_log_loss']:.4f}; winner log loss is "
        f"{ranker['winner_log_loss']:.4f} versus {basic['winner_log_loss']:.4f}. "
        "This experiment does not show a clear predictive gain from the new detailed practice "
        "features. Both use the same detailed-feature DNF model, so this is specifically a "
        "finish-order feature ablation, not an all-pipeline reliability ablation.",
        "",
        "## Scope",
        "",
        "The baseline is a direct ensemble **refit on the expanded stage 2 dataset**, not "
        "the frozen old production artifact. All comparisons use matched development-size "
        "models, known archived entrants and earlier-race-only fitting/calibration. "
        "Historical publication timing and entry-list availability are not reconstructed. "
        "The experiment does not establish improvement over the old production model, "
        "nor validate the full deployment tree configuration. "
        "Monte Carlo convergence across seeds and scenario counts remains untested.",
        "",
        "The main report includes raw metrics, full probability distributions, calibration "
        "plots and global feature importance. See `audit.json` for all probability constraints "
        "and frozen production fingerprints.",
        "",
    ]
    atomic_text_write(study / "assessment.md", "\n".join(assessment))

    # A compact example is a saved walk-forward forecast, not a replay of a model trained later.
    austin = predictions.loc[
        predictions.season.eq(2026) & predictions.race_name.eq("United States Grand Prix")
    ]
    if not austin.empty:
        columns = [
            "driver_id",
            "driver_name",
            "predicted_position",
            "expected_position",
            "p_win",
            "p_podium",
            "p_points",
            "p_dnf",
            "target_finish_position",
        ]
        old = austin.loc[austin.candidate.eq("direct_ensemble"), columns].copy()
        new = austin.loc[austin.candidate.eq("grid_marginal_ensemble"), columns].copy()
        merged = new.merge(
            old.drop(columns="driver_name"),
            on="driver_id",
            suffixes=("_grid", "_direct"),
            validate="one_to_one",
        )
        q = qualifying.loc[
            qualifying.season.eq(2026) & qualifying.race_name.eq("United States Grand Prix")
        ]
        merged = merged.merge(
            q[["driver_id", "expected_quali_position", "p_pole", "label_quali_position"]],
            on="driver_id",
            validate="one_to_one",
        )
        merged = merged.sort_values("predicted_position_grid")
        atomic_text_write(study / "austin_2026.csv", merged.to_csv(index=False))
        lines = [
            "# Austin 2026 — pre-qualifying walk-forward example",
            "",
            "This is the unpromoted development-profile experiment, "
            "not the previous post-quali forecast. "
            "Actual qualifying is included in the CSV for scoring only, "
            "never supplied to the pre-Q forecast.",
            "",
            "| Driver | Expected Q | P(pole) | Direct P(win) | Grid-mixture P(win) | "
            "Grid expected finish | Actual finish |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in merged.itertuples():
            lines.append(
                f"| {row.driver_name} | {row.expected_quali_position:.2f} | {row.p_pole:.1%} | "
                f"{row.p_win_direct:.1%} | {row.p_win_grid:.1%} | "
                f"{row.expected_position_grid:.2f} | {row.target_finish_position_grid} |"
            )
        atomic_text_write(study / "austin_2026.md", "\n".join(lines) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/qualifying_study"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--staging-dir", type=Path, default=Path("work/history_expansion/data"))
    args = parser.parse_args()
    print(
        json.dumps(
            export_checkpoint(
                args.study_dir, DataPaths(args.data_dir), DataPaths(args.staging_dir)
            ),
            indent=2,
        )
    )
