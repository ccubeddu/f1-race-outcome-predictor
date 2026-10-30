"""Read saved stage 5 checkpoints without starting or modifying model training."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table


def checkpoint_progress(study_dir: Path, staging_dir: Path) -> dict:
    protocol = json.loads((study_dir / "protocol.json").read_text())
    cache = staging_dir.parent / "deployment_study" / protocol["run_id"]
    total_per_job = protocol["warmup_races"] + protocol["evaluation_races"]
    jobs = []
    for config in protocol["configurations"]:
        family = "ratings" if config["ratings"] else "base"
        weighting = "uniform" if config["half_life_days"] is None else "decay"
        candidate = f"{family}_{config['start_year']}_{weighting}"
        for mode in protocol["modes"]:
            markers = sorted((cache / mode / candidate).glob("*/complete.json"))
            last_saved = max((path.stat().st_mtime for path in markers), default=None)
            jobs.append(
                {
                    "mode": mode,
                    "candidate": candidate,
                    "completed": len(markers),
                    "total": total_per_job,
                    "last_saved_utc": datetime.fromtimestamp(last_saved, UTC).isoformat()
                    if last_saved is not None
                    else None,
                }
            )
    metrics_path = study_dir / "metrics.json"
    finished = metrics_path.exists() and "simulation_sensitivity" in json.loads(
        metrics_path.read_text()
    )
    return {
        "run_id": protocol["run_id"],
        "completed_fits": sum(row["completed"] for row in jobs),
        "total_fits": total_per_job * len(jobs),
        "jobs": jobs,
        "final_report_present": finished,
        "note": "Checkpoint counts do not prove a process is running. Fit completion excludes "
        "aggregation and simulation-sensitivity checks.",
    }


def completed_metrics(result: dict, staging_dir: Path) -> list[dict]:
    """Report whole completed candidate runs, never selected promising races."""
    import pandas as pd

    from f1predict.evaluation.deployment_study import _metrics

    cache = staging_dir.parent / "deployment_study" / result["run_id"]
    rows = []
    for job in result["jobs"]:
        if job["completed"] != job["total"]:
            continue
        directory = cache / job["mode"] / job["candidate"]
        frames = []
        for marker in sorted(directory.glob("*/complete.json")):
            if json.loads(marker.read_text())["stage"] == "evaluation":
                frames.append(pd.read_parquet(marker.parent / "predictions.parquet"))
        metrics = _metrics(pd.concat(frames, ignore_index=True))
        rows.append({"mode": job["mode"], "candidate": job["candidate"], **metrics})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/deployment_study"))
    parser.add_argument("--staging-dir", type=Path, default=Path("work/history_expansion/data"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="Show whole completed runs only.")
    args = parser.parse_args()
    try:
        result = checkpoint_progress(args.study_dir, args.staging_dir)
    except FileNotFoundError:
        parser.exit(1, "Study protocol missing; start the deployment study first.\n")
    if args.metrics:
        result["completed_metrics"] = completed_metrics(result, args.staging_dir)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        table = Table(title=f"Stage 5 {result['run_id']}: saved fit checkpoints")
        for column in ("Mode", "Candidate", "Completed / total"):
            table.add_column(column)
        for row in result["jobs"]:
            table.add_row(row["mode"], row["candidate"], f"{row['completed']} / {row['total']}")
        console.print(table)
        console.print(f"Fits saved: {result['completed_fits']} / {result['total_fits']}")
        console.print(result["note"])
        if args.metrics:
            metrics_table = Table(title="Completed candidates only — experiment still partial")
            for column in ("Mode", "Candidate", "Races", "Spearman", "Position LL", "Winner LL"):
                metrics_table.add_column(column)
            for row in result["completed_metrics"]:
                metrics_table.add_row(
                    row["mode"],
                    row["candidate"],
                    str(row["races"]),
                    f"{row['spearman']:.4f}",
                    f"{row['position_log_loss']:.4f}",
                    f"{row['winner_log_loss']:.4f}",
                )
            console.print(metrics_table)
            console.print("Selection and the final comparison wait for all configurations.")
