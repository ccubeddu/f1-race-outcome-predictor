"""Reproducible, non-promoting checkpoint for the practice/history feature study."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.history import _fingerprints
from f1predict.data.io import atomic_json_write, atomic_text_write, read_partitioned_table
from f1predict.features.audit import build_feature_audit
from f1predict.features.builder import _base_table
from f1predict.features.practice import (
    _eligible,
    _openf1_table,
    _scheduled_pre_quali_sessions,
    match_practice_drivers,
)


def practice_checkpoint(production: DataPaths, stage: DataPaths, output: Path) -> dict[str, Any]:
    """Audit completed staged matrices; never writes production or trains models."""
    snapshot = json.loads((stage.root / "baseline_snapshot.json").read_text())
    if snapshot["production_root"] != str(production.root.resolve()):
        raise ValueError("Snapshot belongs to a different production dataset")
    unchanged = snapshot["production_fingerprints"] == _fingerprints(production)
    if not unchanged:
        raise ValueError("Production changed after this staging snapshot")
    audit = build_feature_audit(stage)
    pre = pd.read_parquet(stage.features / "pre_quali.parquet")
    baseline = pd.read_parquet(production.features / "pre_quali.parquet")
    manifest = json.loads((stage.features / "manifest.json").read_text())
    roster = _base_table(stage)
    schedule = read_partitioned_table(stage.processed_jolpica, "races")
    sessions = read_partitioned_table(stage.processed_openf1, "practice_sessions")
    eligibility = _scheduled_pre_quali_sessions(schedule, sessions)
    practice = read_partitioned_table(stage.processed_f1db, "practice_results")
    eligible_practice = _eligible(practice, eligibility)
    matched = match_practice_drivers(eligible_practice, roster)
    unmatched = matched.loc[matched["driver_id"].isna()]
    potential_number_join = unmatched.merge(
        roster[["season", "round", "driver_number", "driver_id"]],
        on=["season", "round", "driver_number"],
        how="inner",
        suffixes=("", "_race"),
        validate="many_to_one",
    )
    tables = {}
    for name in ("practice_laps", "practice_stints", "practice_weather"):
        frame = _openf1_table(stage.processed_openf1, name)
        before_cutoff = _eligible(frame, eligibility)
        tables[name] = {
            "raw_processed_rows": len(frame),
            "eligible_rows": len(before_cutoff),
            "excluded_rows": len(frame) - len(before_cutoff),
            "sessions": int(frame["session_key"].nunique()) if not frame.empty else 0,
        }
    rows = []
    for season, group in pre.groupby("season", observed=True):
        old = baseline.loc[baseline["season"].eq(season)]
        rows.append(
            {
                "season": int(season),
                "races": int(group["round"].nunique()),
                "driver_rows": len(group),
                "basic_or_detail_pct": 100 * float(group["practice_data_available"].mean()),
                "legacy_detail_pct": 100 * float(old["openf1_detail_available"].mean())
                if len(old)
                else None,
                "staged_detail_pct": 100 * float(group["openf1_detail_available"].mean()),
                "tyre_long_run_pct": 100 * float(group["practice_long_run_available"].mean()),
            }
        )
    coverage = pd.DataFrame(rows)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "production_unchanged": unchanged,
        "models_trained": False,
        "model_accuracy_improvement_claimed": False,
        "recipe": manifest["practice_recipe"],
        "rows": len(pre),
        "races": int(pre[["season", "round"]].drop_duplicates().shape[0]),
        "pre_features": len(manifest["modes"]["pre_quali"]["features"]),
        "post_features": len(manifest["modes"]["post_quali"]["features"]),
        "practice_classifications": {
            "raw_rows": len(practice),
            "time_eligible_rows": len(eligible_practice),
            "identity_matched_rows": int(matched["driver_id"].notna().sum()),
            "identity_unmatched_rows_excluded": len(unmatched),
            "same_number_wrong_identity_rows_excluded": len(potential_number_join),
            "examples": potential_number_join[
                ["season", "round", "session", "driver_number", "driver_name", "driver_id_race"]
            ]
            .head(30)
            .to_dict("records"),
        },
        "session_timing": {
            "eligible": int(eligibility["is_pre_quali"].sum()),
            "excluded_or_missing": int((~eligibility["is_pre_quali"]).sum()),
            "source_end_timestamp_rows": int(
                eligibility["timing_evidence"].eq("source_session_end").sum()
            ),
        },
        "openf1_tables": tables,
        "coverage": rows,
        "feature_audit": audit,
        "limitations": [
            "Coverage is not predictive accuracy. No model was retrained or promoted.",
            "Before/after coverage changes both acquisition and eligibility/identity rules.",
            "Event-time cutoffs do not reconstruct historical publication/revision timestamps.",
            "Without end timestamps, same-day legacy practice is excluded, including FP3.",
            "Long-run pace is a compound/age/session-relative proxy; "
            "fuel, traffic and flags are not controlled.",
            "A live pre-quali roster defaults provisionally to the previous race; "
            "practice is not an entry list.",
            "2026 remains reused retrospective validation, not an untouched holdout.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_write(output / "audit.json", report)
    atomic_text_write(output / "coverage.csv", coverage.to_csv(index=False))
    atomic_text_write(output / "session_eligibility.csv", eligibility.to_csv(index=False))
    atomic_json_write(output / "feature_manifest.json", manifest)
    lines = [
        "# Practice feature checkpoint — stage 2",
        "",
        f"Built {report['rows']:,} driver-race rows over {report['races']} races; "
        f"{report['pre_features']} pre-quali / {report['post_features']} post-quali features.",
        "",
        "Production fingerprints match the frozen baseline. No model trained or promoted; "
        "no improvement in prediction metrics is claimed.",
        "",
        "## Coverage",
        "",
        "Percentages are driver-race availability, not percentages of all raw practice sessions. "
        "Legacy detail includes the old eligibility rules; staged detail uses the corrected rules.",
        "",
        "| Season | Races | Legacy detail | Staged detail | Tyre-aware long run |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        old = "—" if row["legacy_detail_pct"] is None else f"{row['legacy_detail_pct']:.1f}%"
        lines.append(
            f"| {row['season']} | {row['races']} | {old} | "
            f"{row['staged_detail_pct']:.1f}% | {row['tyre_long_run_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Leakage and identity checks",
        "",
        f"- {len(eligible_practice):,}/{len(practice):,} basic practice rows pass timing; "
        f"{len(unmatched):,} additionally excluded for unmatched driver identity.",
        f"- {len(potential_number_join):,} eligible practice rows share an entrant's number "
        "but not their identity; none are attributed by number.",
        "- Laps, stints and weather share the same completed-session cutoff, with "
        "additional per-record timestamp checks.",
        "- Duplicate identities fail explicitly; missing detail and peer comparisons stay missing.",
        "- Legacy model/input recipe mismatches fail explicitly, preserving old historical replay.",
        "",
        "## Detailed source counts",
        "",
        "| Table | Processed rows | Eligible rows | Excluded rows | Sessions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in tables.items():
        lines.append(
            f"| {name} | {values['raw_processed_rows']:,} | {values['eligible_rows']:,} | "
            f"{values['excluded_rows']:,} | {values['sessions']} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        "## Next approval checkpoint",
        "",
        "Proceed to chronological qualifying-grid modeling and paired feature ablations. "
        "Keep this feature recipe and old production model separate "
        "until backtests justify promotion.",
        "",
        "Sources: [OpenF1 API documentation](https://openf1.org/docs/) and "
        "[free-tier access limits](https://openf1.org/). Bounded season queries are cached "
        "and partitioned into practice sessions only; "
        "requests are paced at least 2.1 seconds apart.",
        "",
    ]
    atomic_text_write(output / "report.md", "\n".join(lines))
    return report
