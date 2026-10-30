"""Verify the full-size study and export its compact decision and Austin examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.history import _fingerprints
from f1predict.data.io import atomic_json_write, atomic_text_write
from f1predict.evaluation.deployment_study import (
    BASELINE,
    KEYS,
    MODES,
    _prediction_error,
    replay_deployment_candidate,
)


def export_checkpoint(output: Path, stage: DataPaths, production: DataPaths) -> dict:
    report = json.loads((output / "metrics.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    frozen = json.loads((stage.root / "baseline_snapshot.json").read_text())
    audit["production_unchanged"] = frozen["production_fingerprints"] == _fingerprints(production)
    cache = stage.root.parent / "deployment_study" / report["protocol"]["run_id"]
    predictions = pd.read_parquet(output / "predictions.parquet")
    dnf_errors = []
    for config in report["protocol"]["configurations"]:
        if not config["ratings"]:
            continue
        suffix = (
            f"{config['start_year']}_{'uniform' if config['half_life_days'] is None else 'decay'}"
        )
        for mode in MODES:
            frame = predictions.loc[predictions["mode"].eq(mode)]
            left = frame.loc[frame.candidate.eq(f"base_{suffix}")].set_index(KEYS).sort_index()
            right = frame.loc[frame.candidate.eq(f"ratings_{suffix}")].set_index(KEYS).sort_index()
            if not left.index.equals(right.index):
                raise AssertionError("Rating ablation has different DNF support")
            dnf_errors.append(float(abs(left.p_dnf - right.p_dnf).max()))
    audit["rating_dnf_control_pairs"] = len(dnf_errors)
    audit["rating_dnf_control_max_error"] = max(dnf_errors, default=0.0)
    contracts = json.loads((cache / "features.json").read_text())
    feature_manifest = json.loads((stage.features / "manifest.json").read_text())
    audit["feature_contract_unchanged"] = all(
        contracts[mode]["base"] == feature_manifest["modes"][mode]["features"] for mode in MODES
    )
    audit["information_cutoffs"] = {
        mode: feature_manifest["modes"][mode]["cutoff"] for mode in MODES
    }
    audit["stage2_matrices_unchanged"] = True
    replay_errors = []
    for mode in MODES:
        original = pd.read_parquet(stage.features / f"{mode}.parquet").sort_values(
            ["race_date", *KEYS], ignore_index=True
        )
        cached = pd.read_parquet(cache / "inputs" / f"{mode}.parquet")
        pd.testing.assert_frame_equal(original, cached[original.columns])
        selected = report["decisions"][mode]["candidate"]
        replay = replay_deployment_candidate(stage, output, mode)
        target_date = pd.Timestamp(replay.race_date.iloc[0])
        saved = predictions.loc[
            predictions["mode"].eq(mode)
            & predictions.candidate.eq(selected)
            & pd.to_datetime(predictions.race_date).eq(target_date)
        ]
        replay_errors.append(_prediction_error(saved, replay))
        example = predictions.loc[
            predictions["mode"].eq(mode)
            & predictions.race_name.eq("United States Grand Prix")
            & predictions.season.eq(2026)
        ]
        if not example.empty:
            _export_austin(example, selected, mode, output)
    audit["selected_cli_replay_max_error"] = max(replay_errors)
    audit["passed"] = (
        audit["passed"]
        and audit["production_unchanged"]
        and audit["stage2_matrices_unchanged"]
        and audit["feature_contract_unchanged"]
        and audit["rating_dnf_control_max_error"] < 1e-10
        and max(replay_errors) < 1e-10
    )
    atomic_json_write(output / "audit.json", audit)
    if not audit["passed"]:
        raise AssertionError("Deployment checkpoint audit failed")
    uncertainty = probability_uncertainty(predictions, report)
    atomic_json_write(output / "probability_uncertainty.json", uncertainty)
    lines = [
        "# Stage 5 assessment",
        "",
        "Matched full-size refits on the staged dataset, not the frozen legacy artifact. "
        "No production promotion. 2023–2025 chooses the candidate; 2026 is reused "
        "retrospective validation, not an untouched holdout.",
        "",
        "Pre-quali means before qualifying starts. Post-quali means after the qualifying "
        "classification **and known starting grid**, not necessarily immediately after Q. "
        "The archive does not establish when late penalties or pit-lane decisions were published.",
        "",
        "## Full-window scores",
        "",
        f"These {report['protocol']['evaluation_races']}-race summaries include selection; "
        "use the separate "
        "2026 results and uncertainty checks before interpreting a gain.",
        "",
        "| Cutoff | Candidate | Spearman ↑ | Top 3 ↑ | Top 10 ↑ | Position LL ↓ | "
        "Winner LL ↓ | DNF Brier ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        selected = report["decisions"][mode]["candidate"]
        for name in dict.fromkeys([BASELINE, selected]):
            values = report["metrics"][mode]["all"][name]
            lines.append(
                f"| {mode} | {name} | {values['spearman']:.4f} | "
                f"{values['top3_accuracy']:.1%} | {values['top10_accuracy']:.1%} | "
                f"{values['position_log_loss']:.4f} | {values['winner_log_loss']:.4f} | "
                f"{values['probabilities']['dnf']['brier']:.5f} |"
            )
    lines += ["", "## Decisions and 2026 check", ""]
    for mode in MODES:
        decision = report["decisions"][mode]
        selected = decision["candidate"]
        base = report["metrics"][mode]["retrospective"][BASELINE]
        values = report["metrics"][mode]["retrospective"][selected]
        lines += [
            f"### {mode}",
            "",
            f"Development-selected: **{selected}**. Recommendation: "
            f"**{decision['recommendation']}**; production unchanged.",
            "",
            f"2026 Spearman {base['spearman']:.4f} → {values['spearman']:.4f}; "
            f"position loss {base['position_log_loss']:.4f} → {values['position_log_loss']:.4f}; "
            f"winner loss {base['winner_log_loss']:.4f} → {values['winner_log_loss']:.4f}.",
            "",
        ]
        failures = [name for name, passed in decision["retrospective_checks"].items() if not passed]
        lines += [
            "Failed 2026 regression guards: " + (", ".join(failures) if failures else "none") + ".",
            "",
        ]
        if selected != BASELINE:
            paired = decision["paired_comparisons"]["retrospective"][selected]["position_log_loss"]
            lo, hi = paired["ci95"]
            lines += [
                f"Paired position-loss delta {paired['candidate_minus_direct']:+.4f}, "
                f"approximate four-race block-bootstrap 95% interval [{lo:+.4f}, {hi:+.4f}].",
                "",
            ]
        lo, hi = report["simulation_sensitivity"][mode]["higher_draw_loss_delta_range"]
        lines += [
            f"At {max(report['protocol']['sensitivity_simulations']):,} draws across three seeds, "
            f"the selected-minus-baseline position-loss range is [{lo:+.4f}, {hi:+.4f}]. "
            "This checks prediction draws only, not calibration-seed or model-fit variation.",
            "",
            f"Features: {len(contracts[mode]['base'])} base / "
            f"{len(contracts[mode]['ratings'])} with ratings and circuit context.",
            "",
        ]
        dnf = uncertainty[mode]["retrospective"]["dnf"]
        lo, hi = dnf["ci95"]
        lines += [
            f"2026 mean per-race DNF Brier delta {dnf['delta']:+.5f}, approximate paired "
            f"95% interval [{lo:+.5f}, {hi:+.5f}]. Races receive equal weight here; "
            "the main table's binary Brier metric weights driver rows equally.",
            "",
        ]
    lines += [
        "## Verification and limitations",
        "",
        "Production fingerprints and staged core feature matrices are unchanged. Training "
        "and calibration cutoffs precede each forecast; loss weights are positive; all "
        "variants share test entrants; probability rows and position columns sum to one. "
        "Saved full-size candidate bundles reproduce their backtest forecasts.",
        "",
        "The full report includes every configuration, per-season metrics in JSON, "
        "calibration plots, uncertainty and feature importance. Coherent probabilities are "
        "not a guarantee of perfect calibration. Source publication timing, archived rosters "
        "and practice/layout/status limitations remain; 2026 has been reused repeatedly.",
        "DNF sampling assumes independent failures across drivers; correlated incidents "
        "and changing race-weather uncertainty are not explicitly simulated.",
        "",
        "The stage 3 qualifying-grid mixture is not stacked with ratings here and has not "
        "passed a full-size promotion comparison. No market advantage is claimed. "
        "Any next production change needs explicit approval, compatible live features "
        "and a genuinely frozen prospective evaluation.",
        "",
        "See `report.md`, `metrics.json`, `audit.json`, `simulation_sensitivity.csv` and "
        "the pre/post-quali `austin_2026_*.md` examples.",
        "",
    ]
    atomic_text_write(output / "assessment.md", "\n".join(lines))
    return audit


def probability_uncertainty(predictions: pd.DataFrame, report: dict) -> dict:
    """Paired four-race blocks, not independent driver rows; no model selection here."""
    result = {}
    for mode in MODES:
        selected = report["decisions"][mode]["candidate"]
        frame = predictions.loc[predictions["mode"].eq(mode)]
        result[mode] = {}
        for split, subset in (("all", frame), ("retrospective", frame.loc[frame.season.ge(2026)])):
            result[mode][split] = {}
            for outcome in ("win", "podium", "points", "dnf"):
                work = subset.assign(
                    error=(subset[f"p_{outcome}"] - subset[f"target_{outcome}"]) ** 2
                )
                scores = (
                    work.groupby(["season", "round", "candidate"]).error.mean().unstack("candidate")
                )
                delta = (scores[selected] - scores[BASELINE]).to_numpy()
                rng = np.random.default_rng(44)
                starts = rng.integers(0, len(delta), size=(2000, int(np.ceil(len(delta) / 4))))
                indices = ((starts[..., None] + np.arange(4)) % len(delta)).reshape(2000, -1)[
                    :, : len(delta)
                ]
                result[mode][split][outcome] = {
                    "delta": float(delta.mean()),
                    "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
                }
    return result


def _export_austin(example: pd.DataFrame, selected: str, mode: str, output: Path) -> None:
    baseline = example.loc[example.candidate.eq(BASELINE)].set_index("driver_id")
    result = example.loc[example.candidate.eq(selected)].copy()
    for name in (
        "p_win",
        "p_podium",
        "p_points",
        "p_dnf",
        "expected_position",
        "predicted_position",
    ):
        result[f"baseline_{name}"] = result.driver_id.map(baseline[name])
    result = result.sort_values("predicted_position")
    atomic_text_write(output / f"austin_2026_{mode}.csv", result.to_csv(index=False))
    lines = [
        f"# Austin 2026 — full-size {mode}",
        "",
        f"Research as-of replay from **{selected}**, selected on 2023–2025. "
        "Not the deployed production model. The entrant set is known retrospectively. "
        "Pre-quali excludes current qualifying; post-quali includes it.",
        "",
        f"Model trained through {str(result.training_end.iloc[0])[:10]}. "
        "Actual positions are for comparison only, not prediction inputs.",
        "",
        "| Rank | Driver | Expected | Baseline win | Selected win | "
        "Podium | Points | DNF | Actual |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples():
        lines.append(
            f"| {row.predicted_position} | {row.driver_name} | {row.expected_position:.2f} | "
            f"{row.baseline_p_win:.1%} | {row.p_win:.1%} | {row.p_podium:.1%} | "
            f"{row.p_points:.1%} | {row.p_dnf:.1%} | {int(row.target_finish_position)} |"
        )
    lines += [
        "",
        "The CSV contains the complete finishing-position distribution and all baseline "
        "event probabilities. Expected-position rank and most likely winner need not match.",
        "",
    ]
    atomic_text_write(output / f"austin_2026_{mode}.md", "\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/deployment_study"))
    parser.add_argument("--staging-dir", type=Path, default=Path("work/history_expansion/data"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(
        json.dumps(
            export_checkpoint(
                args.study_dir, DataPaths(args.staging_dir), DataPaths(args.data_dir)
            ),
            indent=2,
        )
    )
