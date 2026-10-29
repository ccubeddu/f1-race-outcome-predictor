"""Matched chronological ablations for the forward-only rating feature layer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write, atomic_text_write
from f1predict.evaluation.metrics import calibration_table, evaluate_predictions
from f1predict.evaluation.qualifying_study import (
    KINDS,
    _calibrated_dnf,
    _check_baseline,
    _importance,
    _member_forecasts,
    _race_scores,
    load_study_data,
    paired_bootstrap,
)
from f1predict.features.ratings import prepare_rating_features, regulation_era
from f1predict.models.candidates import FinishCandidate, fit_finish_candidates
from f1predict.models.ensemble import ENSEMBLE_WEIGHTS, blend_simulations
from f1predict.models.preprocessing import build_preprocessor, clean_features

NAMES = {
    "direct_ensemble": "Direct ensemble (baseline)",
    "direct_ranker": "Direct ranker",
    "global_ranker": "Global ratings ranker",
    "ratings_ranker": "Ratings + circuit ranker",
    "ratings_ensemble": "Ratings + circuit ensemble",
    "reliability_ranker": "Ratings + circuit + reliability ranker",
    "reliability_ensemble": "Ratings + circuit + reliability ensemble",
}


def load_rating_data(stage: DataPaths, sidecar: Path) -> tuple[pd.DataFrame, dict, dict]:
    data, base = load_study_data(stage)
    ratings, manifest = prepare_rating_features(stage, sidecar)
    data = data.merge(ratings, on=["season", "round", "driver_id"], validate="one_to_one")
    observed = pd.to_datetime(data["asof_race_date"])
    if observed.ge(pd.to_datetime(data["race_date"])).any():
        raise ValueError("A rating snapshot uses a target or future race")
    families = manifest["families"]
    sets = {
        "direct": base,
        "global": [*base, *families["global"]],
        "ratings": [*base, *families["global"], *families["context"]],
    }
    sets["reliability"] = [*sets["ratings"], *families["reliability"]]
    if any(
        c.startswith(("label_", "target_", "post_")) for values in sets.values() for c in values
    ):
        raise ValueError("Rating study exposes unavailable labels as features")
    return data, sets, manifest


def _dnf_inputs(training: pd.DataFrame, features: list[str]) -> FinishCandidate:
    preprocessor = build_preprocessor(training, features)
    preprocessor.fit(clean_features(training, features))
    # The shared DNF helper needs only the preprocessor and declared features.
    return FinishCandidate("dnf_inputs", features, preprocessor, None, False)


def run_rating_study(
    stage: DataPaths,
    production: DataPaths,
    output: Path,
    *,
    start_season: int = 2023,
    end_season: int = 2026,
    profile: str = "backtest",
    simulations: int = 4096,
    warmup_races: int = 12,
    progress: Callable[[str], None] | None = None,
) -> dict:
    _check_baseline(production, stage)
    if profile not in {"full", "backtest"} or simulations < 100 or warmup_races < 5:
        raise ValueError("Invalid rating study settings")
    sidecar = stage.root.parent / "rating_features"
    data, features, rating_manifest = load_rating_data(stage, sidecar)
    races = data[["season", "round", "race_date", "race_name"]].drop_duplicates()
    evaluation = races.loc[races.season.between(start_season, end_season)]
    if evaluation.empty:
        raise ValueError("No evaluation races in requested seasons")
    earlier = races.loc[races.race_date.lt(evaluation.race_date.min())]
    if len(earlier) < warmup_races + 40:
        raise ValueError("Need warmup races and at least 40 earlier training races")
    folds = pd.concat([earlier.tail(warmup_races), evaluation], ignore_index=True)
    params = dict(
        start_season=start_season,
        end_season=end_season,
        profile=profile,
        simulations=simulations,
        warmup_races=warmup_races,
    )
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
    for path in [
        stage.features / "pre_quali.parquet",
        stage.features / "post_quali.parquet",
        sidecar / "ratings.parquet",
        sidecar / "manifest.json",
        *sorted(Path(__file__).parents[1].rglob("*.py")),
    ]:
        digest.update(path.read_bytes())
    run_id = digest.hexdigest()[:16]
    cache = stage.root.parent / "rating_study" / run_id
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json_write(cache / "parameters.json", params)
    temperature_history, dnf_history = {}, {"base": [], "reliability": []}
    outputs, cutoffs = [], []
    for index, event in folds.iterrows():
        fold = cache / f"{int(event['season'])}_{int(event['round']):02d}"
        marker = fold / "complete.json"
        if progress:
            progress(
                f"{index + 1}/{len(folds)}: {event['season']} R{event['round']} "
                f"{event['race_name']} [{'cached' if marker.exists() else 'training'}]"
            )
        if marker.exists():
            state = json.loads(marker.read_text())
            predictions = pd.read_parquet(fold / "predictions.parquet")
        else:
            training = data.loc[
                data.race_date.lt(event.race_date) & data.label_quali_position.gt(0)
            ].copy()
            test = data.loc[data.season.eq(event.season) & data["round"].eq(event["round"])].copy()
            if not training.race_date.max() < test.race_date.min():
                raise AssertionError("Training cutoff violated")
            inference = test[[*features["reliability"]]].copy()
            models = {
                "direct": fit_finish_candidates(
                    training, features["direct"], KINDS, profile=profile
                ),
                "global": fit_finish_candidates(
                    training, features["global"], ("lightgbm_ranker",), profile=profile
                ),
                "ratings": fit_finish_candidates(
                    training, features["ratings"], KINDS, profile=profile
                ),
            }
            raw_base, dnf_base = _calibrated_dnf(
                training, test, models["direct"]["lightgbm_ranker"], dnf_history["base"], profile
            )
            raw_rel, dnf_rel = _calibrated_dnf(
                training,
                test,
                _dnf_inputs(training, features["reliability"]),
                dnf_history["reliability"],
                profile,
            )
            strengths = {
                variant: {kind: model.predict_strength(inference) for kind, model in group.items()}
                for variant, group in models.items()
            }
            strengths["reliability"] = strengths["ratings"]
            records, losses = [], {}
            for variant, scores in strengths.items():
                dnf = dnf_rel if variant == "reliability" else dnf_base
                members, member_losses = _member_forecasts(
                    test,
                    scores,
                    dnf,
                    temperature_history,
                    variant,
                    simulations=simulations,
                    seed=46 + index * 10,
                )
                losses.update(member_losses)
                records.append(members["lightgbm_ranker"].assign(candidate=f"{variant}_ranker"))
                if len(members) > 1:
                    records.append(
                        blend_simulations(members, ENSEMBLE_WEIGHTS["pre_quali"]).assign(
                            candidate=f"{variant}_ensemble"
                        )
                    )
            predictions = pd.concat(records, ignore_index=True)
            predictions["training_end"] = training.race_date.max()
            state = {
                "losses": losses,
                "raw_dnf": {"base": raw_base.tolist(), "reliability": raw_rel.tolist()},
                "actual_dnf": test.target_dnf.astype(int).tolist(),
                "training_end": str(training.race_date.max()),
                "race_date": str(event.race_date),
                "training_rows": len(training),
                "stage": "warmup" if index < warmup_races else "evaluation",
            }
            atomic_parquet_write(predictions, fold / "predictions.parquet")
            for name, group in models.items():
                atomic_parquet_write(
                    _importance(group["lightgbm_ranker"]), fold / f"importance_{name}.parquet"
                )
            atomic_json_write(marker, state)
        if not pd.Timestamp(state["training_end"]) < pd.Timestamp(state["race_date"]):
            raise ValueError("Cached fold violates chronology")
        for key, losses in state["losses"].items():
            temperature_history.setdefault(key, []).append(losses)
        for variant, raw in state["raw_dnf"].items():
            dnf_history[variant].extend(zip(raw, state["actual_dnf"], strict=True))
        cutoffs.append(
            dict(
                season=int(event.season),
                round=int(event["round"]),
                training_end=state["training_end"],
                race_date=state["race_date"],
                training_rows=state["training_rows"],
                stage=state["stage"],
            )
        )
        if index >= warmup_races:
            outputs.append(predictions)
    _check_baseline(production, stage)
    predictions = pd.concat(outputs, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_parquet_write(predictions, output / "predictions.parquet")
    atomic_text_write(output / "cutoffs.csv", pd.DataFrame(cutoffs).to_csv(index=False))
    atomic_json_write(output / "feature_manifest.json", rating_manifest)
    # Store pre-race rating snapshots as interpretable feature evidence, not race labels.
    selected = data.loc[
        data.season.between(start_season, end_season),
        [
            "season",
            "round",
            "driver_id",
            "driver_name",
            "race_name",
            "race_date",
            "asof_race_date",
            *[f for f in features["reliability"] if f.startswith(("rating_", "rel_"))],
        ],
    ]
    atomic_parquet_write(selected, output / "rating_snapshots.parquet")
    last = cache / f"{int(folds.iloc[-1].season)}_{int(folds.iloc[-1]['round']):02d}"
    for name in ("direct", "global", "ratings"):
        atomic_text_write(
            output / f"importance_{name}.csv",
            pd.read_parquet(last / f"importance_{name}.parquet").to_csv(index=False),
        )
    return summarize_rating_study(predictions, output, params | {"run_id": run_id}, rating_manifest)


def summarize_rating_study(
    predictions: pd.DataFrame, output: Path, params: dict, manifest: dict
) -> dict:
    scores = _race_scores(predictions)
    metrics = {}
    for name, group in predictions.groupby("candidate"):
        metrics[name] = evaluate_predictions(group)
        metrics[name]["winner_log_loss"] = float(
            scores.loc[scores.candidate.eq(name), "winner_log_loss"].mean()
        )
    calibration = pd.concat(
        [calibration_table(g).assign(candidate=c) for c, g in predictions.groupby("candidate")],
        ignore_index=True,
    )
    atomic_text_write(output / "race_scores.csv", scores.to_csv(index=False))
    atomic_text_write(output / "calibration.csv", calibration.to_csv(index=False))
    eras = {}
    predictions = predictions.assign(era=predictions.season.map(regulation_era))
    for (name, era), group in predictions.groupby(["candidate", "era"]):
        eras.setdefault(name, {})[era] = evaluate_predictions(group)
    by_season = {}
    for (name, season), group in predictions.groupby(["candidate", "season"]):
        by_season.setdefault(name, {})[str(season)] = evaluate_predictions(group)
    pairings = {c: paired_bootstrap(scores, c) for c in metrics if c != "direct_ensemble"}
    # Additional ranker-only comparisons isolate feature families, not ensemble effects.
    ranker_scores = scores.loc[~scores.candidate.eq("direct_ensemble")].copy()
    ranker_scores.loc[ranker_scores.candidate.eq("direct_ranker"), "candidate"] = "direct_ensemble"
    ablations = {c: paired_bootstrap(ranker_scores, c) for c in ("global_ranker", "ratings_ranker")}
    audit = audit_rating_predictions(predictions)
    if not audit["passed"]:
        raise AssertionError("Rating prediction audit failed")
    atomic_json_write(output / "audit.json", audit)
    result = dict(
        created_at=datetime.now(UTC).isoformat(),
        parameters=params,
        features={name: len(values) for name, values in manifest["families"].items()},
        candidates=metrics,
        paired_comparisons=pairings,
        ranker_ablations=ablations,
        by_era=eras,
        by_season=by_season,
        production_unchanged=True,
        promoted=False,
    )
    atomic_json_write(output / "metrics.json", result)
    from f1predict.evaluation.rating_report import write_rating_report

    write_rating_report(result, calibration, manifest, output)
    return result


def audit_rating_predictions(predictions: pd.DataFrame) -> dict:
    max_row, max_position, max_summary = 0.0, 0.0, 0.0
    supports = []
    for _, candidate in predictions.groupby("candidate"):
        supports.append(
            set(candidate[["season", "round", "driver_id"]].itertuples(index=False, name=None))
        )
        for _, event in candidate.groupby(["season", "round"]):
            size = len(event)
            matrix = event[[f"p_position_{i}" for i in range(1, size + 1)]].to_numpy()
            if not np.isfinite(matrix).all() or matrix.min() < 0:
                raise AssertionError("Invalid position probabilities")
            max_row = max(max_row, float(abs(matrix.sum(axis=1) - 1).max()))
            max_position = max(max_position, float(abs(matrix.sum(axis=0) - 1).max()))
            max_summary = max(
                max_summary,
                float(abs(matrix[:, 0] - event.p_win).max()),
                float(abs(matrix[:, :3].sum(axis=1) - event.p_podium).max()),
                float(abs(matrix[:, : min(10, size)].sum(axis=1) - event.p_points).max()),
            )
    chronology = int(
        pd.to_datetime(predictions.training_end).ge(pd.to_datetime(predictions.race_date)).sum()
    )
    duplicates = int(predictions.duplicated(["candidate", "season", "round", "driver_id"]).sum())
    identical = all(s == supports[0] for s in supports)
    return dict(
        passed=max(max_row, max_position, max_summary) < 1e-8
        and chronology == duplicates == 0
        and identical,
        probability_row_error=max_row,
        position_column_error=max_position,
        summary_probability_error=max_summary,
        chronology_violations=chronology,
        duplicate_keys=duplicates,
        identical_test_support=identical,
    )
