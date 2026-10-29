"""Verify a rating study against the frozen baseline and export a compact assessment."""

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


def export_checkpoint(
    output: Path, stage: DataPaths, production: DataPaths, reference: Path
) -> dict:
    report = json.loads((output / "metrics.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    predictions = pd.read_parquet(output / "predictions.parquet")
    snapshots = pd.read_parquet(output / "rating_snapshots.parquet")
    snapshot = json.loads((stage.root / "baseline_snapshot.json").read_text())
    audit["production_unchanged"] = snapshot["production_fingerprints"] == _fingerprints(production)
    audit["rating_asof_violations"] = int(
        pd.to_datetime(snapshots.asof_race_date).ge(pd.to_datetime(snapshots.race_date)).sum()
    )
    errors = []
    for scope in ("driver", "constructor", "track", "bucket"):
        total = snapshots[
            [f"rel_{scope}_{cause}_rate" for cause in ("mechanical", "incident", "other")]
        ].sum(axis=1)
        errors.append(float(abs(total - snapshots[f"rel_{scope}_dnf_rate"]).max()))
    audit["cause_rate_sum_error"] = max(errors)
    audit["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")
    }
    audit["monte_carlo_convergence_across_seeds_checked"] = False
    audit["stage3_baseline_probability_max_error"] = None
    if (reference / "metrics.json").exists():
        old_report = json.loads((reference / "metrics.json").read_text())
        settings = ("start_season", "end_season", "profile", "simulations", "warmup_races")
        if all(report["parameters"][key] == old_report["parameters"][key] for key in settings):
            old = pd.read_parquet(reference / "predictions.parquet")
            keys = ["season", "round", "driver_id"]
            a = (
                predictions.loc[predictions.candidate.eq("direct_ensemble")]
                .set_index(keys)
                .sort_index()
            )
            b = old.loc[old.candidate.eq("direct_ensemble")].set_index(keys).sort_index()
            if not a.index.equals(b.index):
                raise AssertionError("Stage 3/4 baseline rosters differ")
            columns = [
                "p_win",
                "p_podium",
                "p_points",
                "p_dnf",
                "expected_position",
                *[c for c in a if c.startswith("p_position_")],
            ]
            error = float(
                abs(a[columns].fillna(0).to_numpy() - b[columns].fillna(0).to_numpy()).max()
            )
            audit["stage3_baseline_probability_max_error"] = error
            if error >= 1e-8:
                raise AssertionError("Direct baseline did not reproduce the previous study")
    audit["passed"] = (
        audit["passed"]
        and audit["production_unchanged"]
        and audit["rating_asof_violations"] == 0
        and max(errors) < 1e-10
    )
    atomic_json_write(output / "audit.json", audit)
    if not audit["passed"]:
        raise AssertionError("Rating checkpoint audit failed")

    # A separate paired check for reliability: each unit is one race's mean Brier error.
    work = predictions.assign(squared_dnf_error=(predictions.p_dnf - predictions.target_dnf) ** 2)
    brier = (
        work.groupby(["season", "round", "candidate"]).squared_dnf_error.mean().unstack("candidate")
    )
    delta = (brier.reliability_ensemble - brier.direct_ensemble).to_numpy()
    rng = np.random.default_rng(44)
    starts = rng.integers(0, len(delta), size=(2000, int(np.ceil(len(delta) / 4))))
    indices = ((starts[..., None] + np.arange(4)) % len(delta)).reshape(2000, -1)[:, : len(delta)]
    dnf_interval = {
        "mean_race_brier_delta": float(delta.mean()),
        "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
    }
    atomic_json_write(output / "reliability_uncertainty.json", dnf_interval)

    metrics = report["candidates"]
    base = metrics["direct_ensemble"]
    lines = [
        "# Stage 4 assessment",
        "",
        "Matched pre-qualifying development-profile comparisons on the expanded staged dataset. "
        "Production stays frozen; the old deployed model is not this refitted baseline.",
        "",
        "| Candidate | Spearman ↑ | Position LL ↓ | Winner LL ↓ | DNF Brier ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("direct_ensemble", "Direct baseline"),
        ("ratings_ensemble", "Ratings + circuit"),
        ("reliability_ensemble", "Ratings + circuit + reliability"),
    ):
        m = metrics[name]
        lines.append(
            f"| {label} | {m['spearman']:.4f} | {m['position_log_loss']:.4f} | "
            f"{m['winner_log_loss']:.4f} | {m['probabilities']['dnf']['brier']:.5f} |"
        )
    lines += [
        "",
        "## What changed",
        "",
        "Driver and constructor pairwise ratings have separate qualifying and classified-finisher "
        "channels. Teammate comparisons, retained driver history across moves, discounted team "
        "lineages, layout-specific residuals and coarse geometry buckets are tested explicitly. "
        "A separate DNF variant adds cause-typed reliability histories; it reuses the same finish "
        "models as the circuit variant. No rating reads current-race outcomes.",
        "",
        "## Evidence",
        "",
    ]
    for name, label in (
        ("ratings_ensemble", "Ratings + circuit"),
        ("reliability_ensemble", "Ratings + circuit + reliability"),
    ):
        m = metrics[name]
        change = 100 * (1 - m["position_log_loss"] / base["position_log_loss"])
        delta_win = m["winner_log_loss"] - base["winner_log_loss"]
        lines.append(
            f"- {label}: {change:+.2f}% position-loss reduction; "
            f"winner-loss change {delta_win:+.4f}. Positive loss reduction is better."
        )
    lo, hi = dnf_interval["ci95"]
    lines += [
        f"- Reliability minus baseline mean per-race DNF Brier: "
        f"{dnf_interval['mean_race_brier_delta']:+.5f}, approximate paired block-bootstrap "
        f"95% interval [{lo:+.5f}, {hi:+.5f}]. Negative is better. "
        "The main table averages driver rows; this interval gives each race equal weight.",
        "",
        "Full ranker-only feature ablations, per-season/per-era results, paired intervals, "
        "calibration plots and feature importance are in `report.md` and `metrics.json`. "
        "Do not choose a candidate solely by one metric or one season.",
        "",
        "## Checkpoint decision",
        "",
        "Retain ratings + circuit as a research candidate for stage 5, not a deployment winner. "
        "Inspect the paired rank/winner-loss intervals as well as exact-position loss. "
        "Do not add the separate reliability model to production solely on this experiment.",
        "",
        f"Top-three overlap: {base['top3_accuracy']:.1%} baseline versus "
        f"{metrics['ratings_ensemble']['top3_accuracy']:.1%} with ratings + circuit. "
        f"Top-ten overlap: {base['top10_accuracy']:.1%} versus "
        f"{metrics['ratings_ensemble']['top10_accuracy']:.1%}.",
        "",
        "The stage 3 qualifying-grid mixture remains a separate candidate. These experiments "
        "do not establish that stacking the two additions improves results.",
        "",
        "## Limits and next gate",
        "",
        "These ratings are not uniquely identifiable causal measures of driver skill or car "
        "quality. Source status labels are coarse; unknown retirements are not called mechanical "
        "failures, and incident labels do not prove driver fault. Race-rank ratings are not "
        "fuel/traffic-corrected telemetry pace.",
        "",
        "Only 345/347 layout dates verify; geometry for Turkey 2021 and Las Vegas 2024 is excluded "
        "because dates disagree across sources. Complete pre-race snapshots and their prior-event "
        "dates are saved in `rating_snapshots.parquet`.",
        "",
        "This is reused retrospective validation with a known archived entrant set, not a "
        "prospective forecast. Publication/revision timing and Monte Carlo convergence are "
        "not established. Stage 5 still needs matched full-production settings, training-window "
        "and recency tests, both pre/post-quali cutoffs and a clear promotion decision.",
        "",
    ]
    if "2026+" in report["by_era"]["direct_ensemble"]:
        era_base = report["by_era"]["direct_ensemble"]["2026+"]["spearman"]
        era_ratings = report["by_era"]["ratings_ensemble"]["2026+"]["spearman"]
        lines += [
            f"In the reused 2026 season, Spearman changes from {era_base:.4f} to "
            f"{era_ratings:.4f} with ratings + circuit. Check this era separately "
            "rather than relying on the aggregate score.",
            "",
        ]
    atomic_text_write(output / "assessment.md", "\n".join(lines))

    austin = predictions.loc[
        predictions.season.eq(2026) & predictions.race_name.eq("United States Grand Prix")
    ]
    if not austin.empty:
        cols = [
            "driver_id",
            "predicted_position",
            "expected_position",
            "p_win",
            "p_podium",
            "p_points",
            "p_dnf",
        ]
        merged = austin.loc[
            austin.candidate.eq("direct_ensemble"),
            ["driver_id", "driver_name", "target_finish_position"],
        ]
        for name, prefix in (
            ("direct_ensemble", "direct"),
            ("ratings_ensemble", "ratings"),
            ("reliability_ensemble", "reliability"),
        ):
            part = austin.loc[austin.candidate.eq(name), cols].rename(
                columns={c: f"{prefix}_{c}" for c in cols if c != "driver_id"}
            )
            merged = merged.merge(part, on="driver_id", validate="one_to_one")
        state = snapshots.loc[
            snapshots.season.eq(2026) & snapshots.race_name.eq("United States Grand Prix")
        ]
        merged = merged.merge(
            state[
                [
                    "driver_id",
                    "rating_quali_driver",
                    "rating_quali_constructor",
                    "rating_race_driver",
                    "rating_race_constructor",
                    "asof_race_date",
                ]
            ],
            on="driver_id",
            validate="one_to_one",
        ).sort_values("ratings_predicted_position")
        atomic_text_write(output / "austin_2026.csv", merged.to_csv(index=False))
        example = [
            "# Austin 2026 — stage 4 comparison",
            "",
            "Retrospective **pre-qualifying** forecasts with the archived entrant set. "
            "Current qualifying and race results are not prediction inputs. "
            "This is the development-profile experiment, not a promoted production forecast.",
            "",
            f"Latest rating update: {str(merged.asof_race_date.max())[:10]}. "
            "The actual classified position is shown only for comparison; retirements remain "
            "part of the finishing-order target.",
            "",
            "| Rated order | Driver | Expected position | Baseline win | Rated win | "
            "Rated + reliability win | Rated + reliability DNF | Actual |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in merged.itertuples(index=False):
            example.append(
                f"| {int(row.ratings_predicted_position)} | {row.driver_name} | "
                f"{row.ratings_expected_position:.2f} | {row.direct_p_win:.1%} | "
                f"{row.ratings_p_win:.1%} | {row.reliability_p_win:.1%} | "
                f"{row.reliability_p_dnf:.1%} | {int(row.target_finish_position)} |"
            )
        example += [
            "",
            "The CSV includes all three expected positions, rankings, win/podium/points/DNF "
            "probabilities and the underlying driver/constructor rating snapshot. "
            "Full finishing-position distributions are in `predictions.parquet`.",
            "",
        ]
        atomic_text_write(output / "austin_2026.md", "\n".join(example))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/rating_study"))
    parser.add_argument("--staging-dir", type=Path, default=Path("work/history_expansion/data"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/qualifying_study"))
    args = parser.parse_args()
    print(
        json.dumps(
            export_checkpoint(
                args.study_dir,
                DataPaths(args.staging_dir),
                DataPaths(args.data_dir),
                args.reference,
            ),
            indent=2,
        )
    )
