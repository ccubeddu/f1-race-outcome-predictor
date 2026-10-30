"""Resumable, matched full-size history/recency experiments at both information cutoffs."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write, atomic_text_write
from f1predict.evaluation.metrics import calibration_table, evaluate_predictions
from f1predict.evaluation.qualifying_study import (
    _check_baseline,
    _importance,
    _race_scores,
    paired_bootstrap,
)
from f1predict.evaluation.rating_study import audit_rating_predictions
from f1predict.features.ratings import prepare_rating_features
from f1predict.models.deployment import (
    IDENTIFIERS,
    RECIPE,
    VERSIONS,
    DeploymentConfig,
    calibration_record,
    default_configurations,
    fit_deployment_bundle,
    forecast_scores,
)
from f1predict.models.training import _atomic_joblib_write

KEYS = ["season", "round", "driver_id"]
LABELS = ["target_finish_position", "target_win", "target_podium", "target_points", "target_dnf"]
MODES = ("pre_quali", "post_quali")
BASELINE = "base_2018_uniform"
SELECTION_YEAR = 2026
# Frozen before the run. A lower position loss cannot compensate for large regressions elsewhere.
TOLERANCES = {
    "spearman": 0.005,
    "winner_log_loss": 0.03,
    "top3_accuracy": 0.02,
    "top10_accuracy": 0.02,
    "dnf_brier": 0.001,
}


def _load_inputs(stage: DataPaths) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    manifest = json.loads((stage.features / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Build staged v2 practice features first")
    ratings, rating_manifest = prepare_rating_features(stage, stage.root.parent / "rating_features")
    rating_features = [
        *rating_manifest["families"]["global"],
        *rating_manifest["families"]["context"],
    ]
    frames, features = {}, {}
    for mode in MODES:
        base = manifest["modes"][mode]["features"]
        features[mode] = {"base": base, "ratings": [*base, *rating_features]}
        frames[mode] = (
            pd.read_parquet(stage.features / f"{mode}.parquet")
            .merge(
                ratings[[*KEYS, "asof_race_date", *rating_features]], on=KEYS, validate="one_to_one"
            )
            .sort_values(["race_date", *KEYS], ignore_index=True)
        )
        if (
            pd.to_datetime(frames[mode].asof_race_date)
            .ge(pd.to_datetime(frames[mode].race_date))
            .any()
        ):
            raise ValueError("Rating cutoff violation")
    if not frames[MODES[0]][KEYS].equals(frames[MODES[1]][KEYS]):
        raise ValueError("Pre/post-quali matrices have different entrants")
    return frames, features, rating_manifest


def _prediction_error(a: pd.DataFrame, b: pd.DataFrame) -> float:
    columns = [
        "expected_position",
        "p_win",
        "p_podium",
        "p_points",
        "p_dnf",
        *[c for c in a if c.startswith("p_position_")],
    ]
    a, b = a.set_index("driver_id").sort_index(), b.set_index("driver_id").sort_index()
    if not a.index.equals(b.index):
        raise AssertionError("Serialized bundle changed its roster")
    return float(abs(a[columns].to_numpy() - b[columns].to_numpy()).max())


def _run_configuration(cache_string: str, mode: str, config_dict: dict) -> str:
    """One process owns one candidate's sequential calibration state and checkpoints."""
    cache, config = Path(cache_string), DeploymentConfig(**config_dict)
    options = json.loads((cache / "protocol.json").read_text())
    data = pd.read_parquet(cache / "inputs" / f"{mode}.parquet")
    contracts = json.loads((cache / "features.json").read_text())[mode]
    features = contracts["ratings" if config.ratings else "base"]
    # Ratings are a finish-model ablation, not an unannounced change to DNF inputs.
    dnf_features = contracts["base"]
    folds = pd.read_parquet(cache / "folds.parquet")
    directory = cache / mode / config.name
    history = []
    for index, event in folds.iterrows():
        fold = directory / f"{int(event.season)}_{int(event['round']):02d}"
        marker = fold / "complete.json"
        if index % 10 == 0 or index == len(folds) - 1:
            print(
                f"{mode} {config.name}: {index + 1}/{len(folds)} "
                f"[{'cached' if marker.exists() else 'training'}]",
                flush=True,
            )
        if marker.exists():
            state = json.loads(marker.read_text())
            if (
                state["metadata"]["features"] != features
                or state["metadata"]["dnf_features"] != dnf_features
                or state["metadata"]["configuration"] != config.to_dict()
            ):
                raise ValueError(
                    "Cached model does not match the declared feature/configuration contract"
                )
        else:
            test = data.loc[data.season.eq(event.season) & data["round"].eq(event["round"])].copy()
            cutoff = pd.Timestamp(event.race_date)
            bundle = fit_deployment_bundle(
                data, features, dnf_features, mode, config, cutoff, history
            )
            inference_columns = list(
                dict.fromkeys(
                    [
                        *IDENTIFIERS,
                        *features,
                        *dnf_features,
                        *(["post_quali_position"] if mode == "post_quali" else []),
                    ]
                )
            )
            inference = test[inference_columns].copy()
            strengths, raw = bundle.raw_scores(inference)
            dnf = bundle.calibrate_dnf(raw)
            seed = 44 + int(index) * 10
            forecast = forecast_scores(
                inference,
                strengths,
                dnf,
                bundle.temperatures,
                mode,
                simulations=options["simulations"],
                seed=seed,
            )
            # Targets are attached only after the label-free prediction path has returned.
            prediction = forecast.merge(test[[*KEYS, *LABELS]], on=KEYS, validate="one_to_one")
            prediction = prediction.assign(
                mode=mode, candidate=config.name, training_end=bundle.metadata["training_end"]
            )
            scores = test[
                [*IDENTIFIERS, *LABELS, *(["post_quali_position"] if mode == "post_quali" else [])]
            ].copy()
            for kind, values in strengths.items():
                scores[f"score_{kind}"] = values
            scores["calibrated_dnf"] = dnf
            state = {
                "metadata": bundle.metadata,
                "seed": seed,
                "temperatures": bundle.temperatures,
                "stage": "warmup" if index < options["warmup_races"] else "evaluation",
                "record": calibration_record(
                    test, strengths, raw, dnf, simulations=options["simulations"], seed=seed
                ),
                "serialization_replay_error": None,
            }
            if index == len(folds) - 1:
                model_path = directory / "candidate.joblib"
                _atomic_joblib_write(model_path, bundle)
                replay = joblib.load(model_path).predict(
                    inference, simulations=options["simulations"], seed=seed
                )
                state["serialization_replay_error"] = _prediction_error(forecast, replay)
                if state["serialization_replay_error"] >= 1e-10:
                    raise AssertionError("Saved deployment bundle does not replay the backtest")
                atomic_text_write(
                    directory / "importance.csv",
                    _importance(bundle.models["lightgbm_ranker"]).to_csv(index=False),
                )
                atomic_json_write(directory / "candidate.json", bundle.metadata)
            atomic_parquet_write(prediction, fold / "predictions.parquet")
            atomic_parquet_write(scores, fold / "scores.parquet")
            atomic_json_write(marker, state)
        if not pd.Timestamp(state["metadata"]["training_end"]) < pd.Timestamp(
            state["record"]["race_date"]
        ):
            raise ValueError("Cached training cutoff violation")
        history.append(state["record"])
    return f"{mode}/{config.name}"


def run_deployment_study(
    stage: DataPaths,
    production: DataPaths,
    output: Path,
    *,
    simulations: int = 4096,
    warmup_races: int = 30,
    workers: int = 4,
    progress=None,
) -> dict:
    _check_baseline(production, stage)
    if simulations < 100 or warmup_races < 5 or not 1 <= workers <= 4:
        raise ValueError("Invalid deployment study settings")
    frames, features, ratings = _load_inputs(stage)
    events = frames["pre_quali"][["season", "round", "race_date", "race_name"]].drop_duplicates()
    evaluation = events.loc[events.season.between(2023, 2026)]
    if evaluation.empty or set(evaluation.season) != {2023, 2024, 2025, 2026}:
        raise ValueError("The frozen stage 5 protocol requires completed 2023–2026 races")
    prior = events.loc[events.race_date.lt(evaluation.race_date.min())]
    if len(prior) < warmup_races:
        raise ValueError("Insufficient earlier warmup races")
    folds = pd.concat([prior.tail(warmup_races), evaluation], ignore_index=True)
    configurations = default_configurations()
    protocol = {
        "recipe": RECIPE,
        "profile": "full",
        "versions": VERSIONS,
        "simulations": simulations,
        "warmup_races": warmup_races,
        "calibration_races": 30,
        "modes": list(MODES),
        "configurations": [c.to_dict() for c in configurations],
        "evaluation_races": len(evaluation),
        "evaluation_rows": int(frames["pre_quali"].season.between(2023, 2026).sum()),
        "development": "2023–2025",
        "retrospective_check": "2026 (reused; not an untouched holdout)",
        "baseline": BASELINE,
        "selection": "minimum development position log loss subject to fixed regression tolerances",
        "tolerances": TOLERANCES,
        "production_auto_promotion": False,
        "calibration": "last 30 earlier OOF races; Platt DNF; simulation winner-loss temperatures",
        "ratings": "finish features only; shared base DNF feature contract",
        "training_support": "all archived entrants; no stage 3 qualifying-label training filter",
        "recency": "730.5-day half-life; mean-one loss weights; preprocessing unweighted",
        "legacy_comparison": "deployment-sized v2 refits, not frozen deployed v1 artifacts",
        "qualifying_grid_mixture": "separate research candidate; not stacked or promoted here",
        "sensitivity_simulations": sorted({simulations, simulations * 4}),
        "sensitivity_seeds": [44, 145, 246],
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    for path in [stage.features / f"{mode}.parquet" for mode in MODES] + [
        stage.features / "manifest.json",
        stage.root.parent / "rating_features/ratings.parquet",
        stage.root.parent / "rating_features/manifest.json",
        *sorted(Path(__file__).parents[1].rglob("*.py")),
    ]:
        digest.update(path.read_bytes())
    run_id = digest.hexdigest()[:16]
    cache = stage.root.parent / "deployment_study" / run_id
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json_write(cache / "protocol.json", protocol)
    atomic_json_write(cache / "features.json", features)
    atomic_parquet_write(folds, cache / "folds.parquet")
    for mode in MODES:
        atomic_parquet_write(frames[mode], cache / "inputs" / f"{mode}.parquet")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_write(output / "protocol.json", protocol | {"run_id": run_id})
    atomic_json_write(output / "feature_manifest.json", {"contracts": features, "ratings": ratings})
    if progress:
        progress(
            f"Run {run_id}: {len(folds)} folds × {len(configurations)} configurations × "
            f"2 cutoffs; {workers} one-thread workers"
        )
    jobs = [(mode, config) for config in configurations for mode in MODES]
    if workers == 1:
        for mode, config in jobs:
            _run_configuration(str(cache), mode, config.to_dict())
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(_run_configuration, str(cache), mode, c.to_dict()) for mode, c in jobs
            ]
            for future in as_completed(futures):
                name = future.result()
                if progress:
                    progress(f"Completed {name}")
    _check_baseline(production, stage)
    predictions, cutoff_rows, replay = [], [], []
    for mode, config in jobs:
        directory = cache / mode / config.name
        for event in folds.itertuples(index=False):
            fold = directory / f"{int(event.season)}_{int(event.round):02d}"
            state = json.loads((fold / "complete.json").read_text())
            metadata = state["metadata"]
            cutoff_rows.append(
                {
                    "mode": mode,
                    "candidate": config.name,
                    "season": int(event.season),
                    "round": int(event.round),
                    "stage": state["stage"],
                    "race_date": str(event.race_date),
                    "training_end": metadata["training_end"],
                    "training_start": metadata["training_start"],
                    "training_rows": metadata["training_rows"],
                    "training_races": metadata["training_races"],
                    "calibration_end": metadata["calibration"]["end"],
                    "weight_min": metadata["weight_min"],
                    "weight_max": metadata["weight_max"],
                }
            )
            if state["serialization_replay_error"] is not None:
                replay.append(state["serialization_replay_error"])
            if state["stage"] == "evaluation":
                predictions.append(pd.read_parquet(fold / "predictions.parquet"))
        atomic_text_write(
            output / f"importance_{mode}_{config.name}.csv",
            (directory / "importance.csv").read_text(),
        )
    data = pd.concat(predictions, ignore_index=True)
    audit = {}
    for mode, group in data.groupby("mode"):
        audit[mode] = audit_rating_predictions(group)
    audit["serialization_replay_max_error"] = max(replay)
    audit["production_unchanged"] = True
    cutoffs = pd.DataFrame(cutoff_rows)
    audit["calibration_cutoff_violations"] = int(
        pd.to_datetime(cutoffs.calibration_end).ge(pd.to_datetime(cutoffs.race_date)).sum()
    )
    audit["positive_training_weights"] = bool(cutoffs.weight_min.gt(0).all())
    audit["cross_cutoff_same_entrants"] = all(
        set(data.loc[data["mode"].eq(mode), KEYS].itertuples(index=False, name=None))
        == set(data.loc[data["mode"].eq(MODES[0]), KEYS].itertuples(index=False, name=None))
        for mode in MODES
    )
    audit["source_publication_timing_verified"] = False
    audit["passed"] = (
        all(audit[m]["passed"] for m in MODES)
        and max(replay) < 1e-10
        and audit["calibration_cutoff_violations"] == 0
        and audit["positive_training_weights"]
        and audit["cross_cutoff_same_entrants"]
    )
    atomic_json_write(output / "audit.json", audit)
    if not audit["passed"]:
        raise AssertionError("Deployment study audit failed")
    atomic_parquet_write(data, output / "predictions.parquet")
    atomic_text_write(output / "cutoffs.csv", cutoffs.to_csv(index=False))
    result = summarize_deployment(data, output, protocol | {"run_id": run_id})
    if progress:
        progress("Checking selected/baseline prediction sensitivity across seeds and draw counts")
    result["simulation_sensitivity"] = run_simulation_sensitivity(cache, result, output, workers)
    for mode in MODES:
        consistent = result["simulation_sensitivity"][mode]["higher_draw_loss_gain_all_seeds"]
        result["decisions"][mode]["simulation_gain_consistent"] = consistent
        if not consistent:
            result["decisions"][mode]["recommendation"] = "retain_baseline"
    atomic_json_write(output / "metrics.json", result)
    from f1predict.evaluation.deployment_report import write_deployment_report

    write_deployment_report(result, pd.read_csv(output / "calibration.csv"), output)
    _check_baseline(production, stage)
    return result


def _metrics(frame: pd.DataFrame) -> dict:
    metrics = evaluate_predictions(frame)
    metrics["winner_log_loss"] = float(
        _race_scores(frame.assign(candidate="score")).winner_log_loss.mean()
    )
    return metrics


def regression_checks(candidate: dict, baseline: dict) -> dict[str, bool]:
    return {
        "spearman": candidate["spearman"] >= baseline["spearman"] - TOLERANCES["spearman"],
        "winner_loss": candidate["winner_log_loss"]
        <= baseline["winner_log_loss"] + TOLERANCES["winner_log_loss"],
        "top3": candidate["top3_accuracy"]
        >= baseline["top3_accuracy"] - TOLERANCES["top3_accuracy"],
        "top10": candidate["top10_accuracy"]
        >= baseline["top10_accuracy"] - TOLERANCES["top10_accuracy"],
        "dnf_brier": candidate["probabilities"]["dnf"]["brier"]
        <= baseline["probabilities"]["dnf"]["brier"] + TOLERANCES["dnf_brier"],
    }


def select_configuration(development: dict) -> dict:
    baseline = development[BASELINE]
    checks = {name: regression_checks(values, baseline) for name, values in development.items()}
    eligible = [
        name
        for name, values in development.items()
        if all(checks[name].values())
        and values["position_log_loss"] <= baseline["position_log_loss"]
    ]
    selected = min(eligible, key=lambda name: (development[name]["position_log_loss"], name))
    return {
        "candidate": selected,
        "development_checks": checks,
        "selection_year": SELECTION_YEAR,
        "baseline_fallback": selected == BASELINE,
        "production_promoted": False,
    }


def summarize_deployment(data: pd.DataFrame, output: Path, protocol: dict) -> dict:
    reports, calibrations, race_scores, decisions = {}, [], [], {}
    for mode, frame in data.groupby("mode"):
        reports[mode] = {}
        for split, subset in (
            ("all", frame),
            ("development", frame.loc[frame.season.lt(SELECTION_YEAR)]),
            ("retrospective", frame.loc[frame.season.ge(SELECTION_YEAR)]),
        ):
            reports[mode][split] = {
                name: _metrics(group) for name, group in subset.groupby("candidate")
            }
        reports[mode]["by_season"] = {
            str(season): {
                name: _metrics(group) for name, group in season_frame.groupby("candidate")
            }
            for season, season_frame in frame.groupby("season")
        }
        decisions[mode] = select_configuration(reports[mode]["development"])
        selected = decisions[mode]["candidate"]
        checks = regression_checks(
            reports[mode]["retrospective"][selected], reports[mode]["retrospective"][BASELINE]
        )
        decisions[mode]["retrospective_checks"] = checks
        comparisons = {}
        for split, subset in (
            ("all", frame),
            ("retrospective", frame.loc[frame.season.ge(SELECTION_YEAR)]),
        ):
            scores = _race_scores(subset)
            scores.loc[scores.candidate.eq(BASELINE), "candidate"] = "direct_ensemble"
            comparisons[split] = {
                name: paired_bootstrap(scores, name)
                for name in subset.candidate.unique()
                if name != BASELINE
            }
        decisions[mode]["paired_comparisons"] = comparisons
        ci_passed = (
            selected != BASELINE
            and comparisons["retrospective"][selected]["position_log_loss"]["ci95"][1] < 0
        )
        decisions[mode]["retrospective_position_interval_below_zero"] = ci_passed
        decisions[mode]["recommendation"] = (
            "prospective_trial_only" if ci_passed and all(checks.values()) else "retain_baseline"
        )
        calibrations += [
            calibration_table(g).assign(mode=mode, candidate=c)
            for c, g in frame.groupby("candidate")
        ]
        race_scores.append(_race_scores(frame).assign(mode=mode))
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "metrics": reports,
        "decisions": decisions,
        "production_promoted": False,
        "production_unchanged": True,
    }
    atomic_json_write(output / "metrics.json", result)
    atomic_text_write(output / "calibration.csv", pd.concat(calibrations).to_csv(index=False))
    atomic_text_write(output / "race_scores.csv", pd.concat(race_scores).to_csv(index=False))
    from f1predict.evaluation.deployment_report import write_deployment_report

    write_deployment_report(result, pd.concat(calibrations), output)
    return result


def replay_deployment_candidate(
    stage: DataPaths,
    study: Path,
    mode: str,
    candidate: str = "selected",
    *,
    simulations: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Replay the final stored as-of bundle, never a model trained on that target race."""
    report = json.loads((study / "metrics.json").read_text())
    if mode not in MODES:
        raise ValueError("Unknown prediction cutoff")
    if candidate == "selected":
        candidate = report["decisions"][mode]["candidate"]
    configs = {DeploymentConfig(**c).name for c in report["protocol"]["configurations"]}
    if candidate not in configs:
        raise ValueError("Unknown candidate")
    cache = stage.root.parent / "deployment_study" / report["protocol"]["run_id"]
    directory = cache / mode / candidate
    bundle = joblib.load(directory / "candidate.joblib")
    data = pd.read_parquet(cache / "inputs" / f"{mode}.parquet")
    frame = data.loc[
        pd.to_datetime(data.race_date).eq(pd.Timestamp(bundle.metadata["target_date"]))
    ]
    fold = directory / f"{int(frame.season.iloc[0])}_{int(frame['round'].iloc[0]):02d}"
    state = json.loads((fold / "complete.json").read_text())
    return bundle.predict(
        frame,
        simulations=simulations if simulations is not None else report["protocol"]["simulations"],
        seed=seed if seed is not None else state["seed"],
    )


def _sensitivity_job(
    cache_string: str, mode: str, candidate: str, simulations: int, seed: int
) -> dict:
    cache = Path(cache_string)
    folds = pd.read_parquet(cache / "folds.parquet")
    frames = []
    for index, event in folds.loc[folds.season.ge(SELECTION_YEAR)].iterrows():
        fold = cache / mode / candidate / f"{int(event.season)}_{int(event['round']):02d}"
        state = json.loads((fold / "complete.json").read_text())
        scores = pd.read_parquet(fold / "scores.parquet")
        strengths = {kind: scores[f"score_{kind}"].to_numpy() for kind in state["temperatures"]}
        forecast = forecast_scores(
            scores,
            strengths,
            scores.calibrated_dnf.to_numpy(),
            state["temperatures"],
            mode,
            simulations=simulations,
            seed=seed + int(index) * 10,
        )
        frames.append(forecast.merge(scores[[*KEYS, *LABELS]], on=KEYS, validate="one_to_one"))
    metrics = _metrics(pd.concat(frames, ignore_index=True))
    return {
        "mode": mode,
        "candidate": candidate,
        "simulations": simulations,
        "seed": seed,
        **{
            key: metrics[key]
            for key in (
                "spearman",
                "position_log_loss",
                "winner_log_loss",
                "top3_accuracy",
                "top10_accuracy",
            )
        },
    }


def run_simulation_sensitivity(cache: Path, report: dict, output: Path, workers: int) -> dict:
    protocol = report["protocol"]
    jobs = [
        (str(cache), mode, candidate, simulations, seed)
        for mode in MODES
        for candidate in sorted({BASELINE, report["decisions"][mode]["candidate"]})
        for simulations in protocol["sensitivity_simulations"]
        for seed in protocol["sensitivity_seeds"]
    ]
    if workers == 1:
        rows = [_sensitivity_job(*args) for args in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = list(pool.map(_sensitivity_job_unpack, jobs))
    frame = pd.DataFrame(rows).sort_values(["mode", "candidate", "simulations", "seed"])
    atomic_text_write(output / "simulation_sensitivity.csv", frame.to_csv(index=False))
    summaries = {}
    for mode in MODES:
        selected = report["decisions"][mode]["candidate"]
        scores = frame.loc[frame["mode"].eq(mode)].pivot(
            index=["simulations", "seed"], columns="candidate", values="position_log_loss"
        )
        delta = scores[selected] - scores[BASELINE]
        high = delta.loc[max(protocol["sensitivity_simulations"])].to_numpy()
        summaries[mode] = {
            "selected": selected,
            "higher_draw_loss_delta_range": [float(high.min()), float(high.max())],
            "higher_draw_loss_gain_all_seeds": bool(selected != BASELINE and (high < 0).all()),
            "scope": "2026 prediction-draw sensitivity only; calibration and model fits held fixed",
        }
    return summaries


def _sensitivity_job_unpack(args: tuple) -> dict:
    return _sensitivity_job(*args)
