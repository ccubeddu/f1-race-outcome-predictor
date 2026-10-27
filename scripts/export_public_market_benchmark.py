"""Export archived public market probes and score the retrospective entrant grid.

Run from the repository root. Replays checked-in raw snapshots, seeding missing
archives from work/ probes on first export. No network requests are made here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.bookmakers import (
    compare_winner_market,
    load_odds,
    score_history,
    score_winner_market,
)
from f1predict.data.io import atomic_json_write
from f1predict.data.polymarket import select_snapshot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "public_market_benchmark"
RACES = [
    ("azerbaijan", 15, "2026-09-26T11:00:00Z", "2026-09-25T12:00:00Z"),
    ("bahrain", 16, "2026-10-04T07:00:00Z", "2026-10-03T08:00:00Z"),
    ("singapore", 17, "2026-10-11T12:00:00Z", "2026-10-10T13:00:00Z"),
    ("united-states", 18, "2026-10-25T20:00:00Z", "2026-10-24T21:00:00Z"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    exclusions: list[dict] = []
    for mode, source_name, candidate in (
        ("pre_quali", "polymarket_pre_quali", "anchored_ensemble"),
        ("post_quali", "polymarket_snapshots", "pole_aware_ensemble"),
    ):
        source = ROOT / "work" / source_name
        predictions_path = ROOT / "data" / "evaluation" / mode / "predictions.parquet"
        predictions = pd.read_parquet(predictions_path)
        records = []
        manifest = []
        for slug, round_, race_start, qualifying_start in RACES:
            path = OUT / mode / slug
            path.mkdir(parents=True, exist_ok=True)
            for suffix in ("raw.json", "mapping.json"):
                probe = source / f"{slug}_{suffix}"
                if probe.exists() and not (path / suffix).exists():
                    shutil.copyfile(probe, path / suffix)
            race = predictions.loc[
                predictions["season"].eq(2026)
                & predictions["round"].eq(round_)
                & predictions["candidate"].eq(candidate)
            ].copy()
            raw = json.loads((path / "raw.json").read_text())
            mapping = json.loads((path / "mapping.json").read_text())
            cutoff = (
                datetime.fromisoformat(qualifying_start.replace("Z", "+00:00"))
                - timedelta(minutes=15)
                if mode == "pre_quali"
                else datetime.fromisoformat(race_start.replace("Z", "+00:00")) - timedelta(hours=2)
            )
            try:
                snapshot = select_snapshot(
                    raw["event"],
                    raw["histories"],
                    driver_markets=mapping,
                    expected_driver_ids=race["driver_id"].tolist(),
                    cutoff=cutoff,
                    max_age_seconds=300,
                )
            except ValueError as exc:
                exclusions.append(
                    {
                        "mode": mode,
                        "event": raw["event"]["slug"],
                        "event_id": raw["event"]["id"],
                        "status": "rejected",
                        "error": str(exc),
                    }
                )
                continue
            metadata = snapshot.metadata
            previous_metadata = source / f"{slug}_metadata.json"
            if not previous_metadata.exists():
                previous_metadata = path / "metadata.json"
            if previous_metadata.exists():
                previous = json.loads(previous_metadata.read_text())
                metadata["retrieved_at"] = previous.get("retrieved_at")
            snapshot.quotes.to_csv(path / "quotes.csv", index=False)
            race.to_csv(path / "prediction.csv", index=False)
            compared = compare_winner_market(race, load_odds(path / "quotes.csv"))
            compared.to_csv(path / "comparison.csv", index=False)
            winner = str(race.loc[race["target_win"].eq(1), "driver_id"].iloc[0])
            score = score_winner_market(compared, winner)
            model_favorite = compared.loc[compared["p_win"].idxmax(), "driver_name"]
            market_favorite = compared.loc[compared["market_probability"].idxmax(), "driver_name"]
            winning_row = compared.loc[compared["driver_id"].eq(winner)].iloc[0]
            record = {
                "race": race["race_name"].iloc[0],
                "mode": mode,
                "round": round_,
                "winner": score["winner"],
                "model_favorite": model_favorite,
                "market_favorite": market_favorite,
                "model_correct": model_favorite == score["winner"],
                "market_correct": market_favorite == score["winner"],
                "model_winner_probability": winning_row["p_win"],
                "market_winner_probability": winning_row["market_probability"],
                **score,
                "cutoff_at": metadata["cutoff_at"],
                "max_quote_age_seconds": pd.read_csv(path / "quotes.csv")[
                    "quote_age_seconds"
                ].max(),
                "unmatched_observed_probability_sum": metadata[
                    "unmatched_observed_probability_sum"
                ],
                "unmatched_missing_histories": metadata["unmatched_missing_histories"],
            }
            records.append(record)
            metadata.update(
                {
                    "raw_sha256": sha256(path / "raw.json"),
                    "prediction_sha256": sha256(path / "prediction.csv"),
                    "backtest_source_sha256": sha256(predictions_path),
                    "model_candidate": candidate,
                    "model_training_end": race["training_end"].unique().tolist(),
                    "analysis_kind": (
                        "retrospective; normalized conditional on supplied actual entrants"
                    ),
                    "feature_publication_times_verified": False,
                    "model_selection_untouched": False,
                    "backtest_profile": (
                        "stored reduced training profile; differs from full production fit"
                    ),
                    "race_start_source": "local Jolpica race schedule; scheduled start time",
                    "race_start_at": race_start,
                    "qualifying_start_at": qualifying_start,
                }
            )
            atomic_json_write(path / "metadata.json", metadata)
            manifest.append(
                {
                    "race": record["race"],
                    "season": 2026,
                    "round": round_,
                    "mode": mode,
                    "prediction_csv": str((path / "prediction.csv").relative_to(OUT)),
                    "odds_csv": str((path / "quotes.csv").relative_to(OUT)),
                    "winner": winner,
                    "model_cutoff_at": metadata["cutoff_at"],
                    # Publication times for the historic feature inputs and actual
                    # qualifying completion are absent. Preserve unknowns explicitly.
                    "features_available_at": None,
                    "qualifying_end_at": None,
                    "odds_snapshot_at": metadata["cutoff_at"],
                    "qualifying_start_at": qualifying_start,
                    "race_start_at": race_start,
                    "source_url": metadata["source_url"],
                    "source_market_id": metadata["event_id"],
                    "source_kind": "prediction_market",
                    "timing_evidence": "CLOB timestamped raw responses and hashes in race metadata",
                }
            )
        frame = pd.DataFrame.from_records(records)
        frame.to_csv(OUT / f"{mode}_scores.csv", index=False)
        manifest_path = OUT / f"{mode}_manifest.csv"
        pd.DataFrame.from_records(manifest).to_csv(manifest_path, index=False)
        audited = score_history(manifest_path)
        audited.to_csv(OUT / f"{mode}_alignment.csv", index=False)
        differences = (frame["model_log_loss"] - frame["market_log_loss"]).to_numpy()
        n = len(differences)
        # Enumerate all n^n resamples for this small paired sample. Interval is
        # descriptive; it does not adjust for repeated model selection.
        bootstrap = [
            np.mean(differences[list(indices)]) for indices in itertools.product(range(n), repeat=n)
        ]
        reports[mode] = {
            "races": n,
            "model_correct": int(frame["model_correct"].sum()),
            "market_correct": int(frame["market_correct"].sum()),
            "model_mean_log_loss": float(frame["model_log_loss"].mean()),
            "market_mean_log_loss": float(frame["market_log_loss"].mean()),
            "model_mean_brier": float(frame["model_brier"].mean()),
            "market_mean_brier": float(frame["market_brier"].mean()),
            "model_better_log_loss_races": int((differences < 0).sum()),
            "model_minus_market_mean_log_loss": float(differences.mean()),
            "paired_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "timing_eligible_races": int(audited["timing_eligible"].sum()),
        }
    atomic_json_write(
        OUT / "metrics.json",
        {
            "source": "Polymarket prediction market, separate from bookmaker comparison",
            "analysis_kind": "retrospective, conditional on the actual entrant grid",
            "cutoff_policy": {
                "pre_quali": "scheduled qualifying start minus 15 minutes",
                "post_quali": "scheduled race start minus 2 hours",
            },
            "max_quote_age_seconds": 300,
            "modes": reports,
            "exclusions": exclusions,
        },
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    export()
