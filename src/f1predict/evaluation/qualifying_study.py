"""Resumable chronological qualifying -> race experiment; no production promotion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from f1predict.config import DataPaths
from f1predict.data.history import _fingerprints
from f1predict.data.io import atomic_json_write, atomic_parquet_write, atomic_text_write
from f1predict.evaluation.grid_simulation import position_matrix, prediction_frame
from f1predict.evaluation.metrics import calibration_table, evaluate_predictions
from f1predict.models.candidates import FinishCandidate, fit_finish_candidates
from f1predict.models.ensemble import ENSEMBLE_WEIGHTS, blend_simulations
from f1predict.models.preprocessing import clean_features
from f1predict.models.qualifying import (
    CONDITIONAL_FEATURES,
    GridForecast,
    choose_qualifying_temperature,
    conditional_inputs,
    fit_qualifying_model,
    sample_grid_positions,
    scenario_strengths,
)
from f1predict.models.training import build_dnf_classifier

TEMPERATURES = (0.3, 0.45, 0.65, 0.85, 1.1, 1.45, 1.9)
KINDS = tuple(ENSEMBLE_WEIGHTS["pre_quali"])
LABELS = ["target_finish_position", "target_win", "target_podium", "target_points", "target_dnf"]
KEYS = ["season", "round", "driver_id"]
IDENTIFIERS = [*KEYS, "race_date", "race_name", "driver_name", "constructor_name"]


def load_study_data(paths: DataPaths) -> tuple[pd.DataFrame, list[str]]:
    manifest = json.loads((paths.features / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Build the staged v2 feature recipe first")
    features = manifest["modes"]["pre_quali"]["features"]
    if any(c.startswith(("post_", "target_", "label_", "scenario_")) for c in features):
        raise ValueError("Pre-quali feature contract exposes unavailable information")
    data = pd.read_parquet(paths.features / "pre_quali.parquet")
    post = pd.read_parquet(paths.features / "post_quali.parquet")
    # Kept explicitly as a LABEL, not an input to either pre-Q forecast stage.
    qlabels = post[[*KEYS, "post_quali_position"]].rename(
        columns={"post_quali_position": "label_quali_position"}
    )
    data = data.merge(qlabels, on=KEYS, how="left", validate="one_to_one")
    return data.sort_values(["race_date", *KEYS], ignore_index=True), features


def _calibrated_dnf(
    training: pd.DataFrame,
    test: pd.DataFrame,
    candidate: FinishCandidate,
    history: list[tuple[float, int]],
    profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = build_dnf_classifier(profile=profile)
    model.fit(
        candidate.preprocessor.transform(clean_features(training, candidate.features)),
        training["target_dnf"].to_numpy(dtype=int),
    )
    raw = model.predict_proba(
        candidate.preprocessor.transform(clean_features(test, candidate.features))
    )[:, 1]
    calibrated = raw
    if len(history) >= 50:
        old, actual = np.asarray(history).T
        if np.unique(actual).size == 2 and np.unique(old).size > 1:
            old = np.clip(old, 1e-6, 1 - 1e-6)
            calibrator = LogisticRegression(random_state=44).fit(
                np.log(old / (1 - old))[:, None], actual.astype(int)
            )
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            calibrated = calibrator.predict_proba(np.log(clipped / (1 - clipped))[:, None])[:, 1]
    return raw, np.clip(calibrated, 0.001, 0.999)


def _temperature(history: list[list[float]]) -> float:
    return TEMPERATURES[int(np.argmin(np.mean(history, axis=0)))] if len(history) >= 5 else 0.85


def _member_forecasts(
    test: pd.DataFrame,
    strengths: dict[str, np.ndarray],
    dnf: np.ndarray,
    histories: dict[str, list[list[float]]],
    prefix: str,
    *,
    simulations: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[float]]]:
    """Choose each temperature BEFORE consulting this race's targets."""
    members, losses = {}, {}
    identifiers = test[[*IDENTIFIERS, *LABELS]].copy()
    winner = test["target_win"].to_numpy(dtype=bool)
    for kind, strength in strengths.items():
        key = f"{prefix}/{kind}"
        selected = _temperature(histories.get(key, []))
        probabilities = {
            t: position_matrix(strength, dnf, simulations=simulations, temperature=t, seed=seed)
            for t in TEMPERATURES
        }
        members[kind] = prediction_frame(identifiers, probabilities[selected], dnf)
        members[kind]["race_temperature"] = selected
        # Scores are appended to calibration state only AFTER all predictions are frozen.
        losses[key] = [-float(np.log(matrix[winner, 0][0])) for matrix in probabilities.values()]
    return members, losses


def _importance(model: FinishCandidate) -> pd.DataFrame:
    names = model.preprocessor.get_feature_names_out()
    categories = model.preprocessor.transformers_[1][2]
    sources = []
    for name in names:
        clean = name.split("__", 1)[-1]
        sources.append(
            next((c for c in categories if clean == c or clean.startswith(f"{c}_")), clean)
        )
    frame = pd.DataFrame(
        {
            "feature": sources,
            "gain": model.model.booster_.feature_importance(importance_type="gain"),
        }
    )
    frame = frame.groupby("feature", as_index=False)["gain"].sum()
    frame["importance"] = frame["gain"] / max(float(frame["gain"].sum()), 1)
    return frame.sort_values("gain", ascending=False, ignore_index=True)


def _check_baseline(production: DataPaths, stage: DataPaths) -> None:
    if stage.root.resolve() == production.root.resolve():
        raise ValueError("This study must run in the isolated staging dataset")
    snapshot = json.loads((stage.root / "baseline_snapshot.json").read_text())
    if snapshot["production_root"] != str(production.root.resolve()) or snapshot[
        "production_fingerprints"
    ] != _fingerprints(production):
        raise ValueError("Frozen production baseline changed; use a new staging snapshot")


def run_qualifying_study(
    stage: DataPaths,
    production: DataPaths,
    output: Path,
    *,
    start_season: int = 2023,
    end_season: int = 2026,
    profile: str = "backtest",
    grid_scenarios: int = 256,
    simulations: int = 4096,
    warmup_races: int = 12,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Expand/train before every race, with sequential held-out calibration state.

    Conditional training uses actual Q only in past races to learn P(Y|X,Q).
    Inference integrates P(Y|X,Q) P(Q|X); no predicted Q is used as an in-sample
    training covariate. Every Q forecast used for calibration is strictly OOF.
    """
    _check_baseline(production, stage)
    if profile not in {"full", "backtest"} or warmup_races < 5 or grid_scenarios < 32:
        raise ValueError("Invalid study profile, warmup or scenario count")
    if simulations < 100 or simulations % grid_scenarios:
        raise ValueError("Race simulations must be >=100 and a multiple of grid scenarios")
    data, features = load_study_data(stage)
    races = data[["season", "round", "race_date", "race_name"]].drop_duplicates()
    evaluation = races.loc[races["season"].between(start_season, end_season)]
    if evaluation.empty:
        raise ValueError("No evaluation races in the requested range")
    prior = races.loc[races["race_date"].lt(evaluation["race_date"].min())]
    if len(prior) < warmup_races + 40:
        raise ValueError("Require warmup plus at least 40 earlier training races")
    folds = pd.concat([prior.tail(warmup_races), evaluation], ignore_index=True)
    params = dict(
        start_season=start_season,
        end_season=end_season,
        profile=profile,
        grid_scenarios=grid_scenarios,
        simulations=simulations,
        warmup_races=warmup_races,
    )
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
    for path in [
        stage.features / "pre_quali.parquet",
        stage.features / "post_quali.parquet",
        stage.features / "manifest.json",
        *sorted(Path(__file__).parents[1].rglob("*.py")),
    ]:
        digest.update(path.read_bytes())
    run_id = digest.hexdigest()[:16]
    cache = stage.root.parent / "qualifying_study" / run_id
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json_write(cache / "parameters.json", params)
    histories: dict[str, list[list[float]]] = {}
    qhistory: list[tuple[np.ndarray, np.ndarray]] = []
    dnf_history: list[tuple[float, int]] = []
    outputs, qualifying_outputs, cutoffs = [], [], []
    for index, event in folds.iterrows():
        fold = cache / f"{int(event['season'])}_{int(event['round']):02d}"
        label = (
            f"{index + 1}/{len(folds)}: {event['season']} R{event['round']} {event['race_name']}"
        )
        if progress:
            progress(label + (" [cached]" if (fold / "complete.json").exists() else " [training]"))
        if (fold / "complete.json").exists():
            state = json.loads((fold / "complete.json").read_text())
            predictions = pd.read_parquet(fold / "race.parquet")
            qforecast = pd.read_parquet(fold / "qualifying.parquet")
        else:
            training = data.loc[data["race_date"].lt(event["race_date"])].copy()
            # Shared race-training support, so missing Q labels don't give one model
            # different training races/rows. Features remain from before each past Q.
            training = training.loc[training["label_quali_position"].gt(0)].copy()
            test = data.loc[
                data["season"].eq(event["season"]) & data["round"].eq(event["round"])
            ].copy()
            if not training["race_date"].max() < test["race_date"].min():
                raise AssertionError("Chronological training cutoff violated")
            # The actual current Q label and race outcome never reach either scoring call.
            inference = test[
                [*KEYS, "race_date", "constructor_id", *[f for f in features if f not in KEYS]]
            ].copy()
            qmodel = fit_qualifying_model(training, features, profile=profile)
            qscores = qmodel.predict_strength(inference)
            qtemperature = choose_qualifying_temperature(qhistory)
            seed = 44 + index * 10
            grids = sample_grid_positions(
                qscores, scenarios=grid_scenarios, temperature=qtemperature, seed=seed
            )
            qdraws = sample_grid_positions(
                qscores, scenarios=8192, temperature=qtemperature, seed=seed + 1
            )
            qforecast = GridForecast.from_positions(qdraws).to_frame(
                test[[*IDENTIFIERS, "label_quali_position"]]
            )
            qforecast["qualifying_temperature"] = qtemperature
            qforecast["training_end"] = training["race_date"].max()
            direct = fit_finish_candidates(training, features, KINDS, profile=profile)
            conditional = conditional_inputs(training, training["label_quali_position"].to_numpy())
            grid_models = fit_finish_candidates(
                conditional, [*features, *CONDITIONAL_FEATURES], KINDS, profile=profile
            )
            basic_features = [
                f
                for f in features
                if not f.startswith("openf1_")
                and f not in {"practice_long_run_available", "practice_data_available"}
            ]
            basic = fit_finish_candidates(
                training, basic_features, ("lightgbm_ranker",), profile=profile
            )
            raw_dnf, dnf = _calibrated_dnf(
                training, test, direct["lightgbm_ranker"], dnf_history, profile
            )
            variant_strengths = {
                "direct": {k: m.predict_strength(inference) for k, m in direct.items()},
                "grid_marginal": {
                    k: scenario_strengths(m, inference, grids) for k, m in grid_models.items()
                },
                "basic_practice": {k: m.predict_strength(inference) for k, m in basic.items()},
            }
            records, losses = [], {}
            for variant, strengths in variant_strengths.items():
                members, member_losses = _member_forecasts(
                    test, strengths, dnf, histories, variant, simulations=simulations, seed=seed + 2
                )
                losses.update(member_losses)
                anchor = members["lightgbm_ranker"].assign(candidate=f"{variant}_ranker")
                records.append(anchor)
                if len(members) > 1:
                    records.append(
                        blend_simulations(members, ENSEMBLE_WEIGHTS["pre_quali"]).assign(
                            candidate=f"{variant}_ensemble"
                        )
                    )
            predictions = pd.concat(records, ignore_index=True)
            predictions["training_end"] = training["race_date"].max()
            state = {
                "losses": losses,
                "q_scores": qscores.tolist(),
                "q_positions": test["label_quali_position"].astype(float).tolist(),
                "raw_dnf": raw_dnf.tolist(),
                "actual_dnf": test["target_dnf"].astype(int).tolist(),
                "training_end": str(training["race_date"].max()),
                "training_rows": len(training),
                "prediction_date": str(event["race_date"]),
                "stage": "warmup" if index < warmup_races else "evaluation",
            }
            atomic_parquet_write(predictions, fold / "race.parquet")
            atomic_parquet_write(qforecast, fold / "qualifying.parquet")
            for name, model in (
                ("qualifying", qmodel),
                ("direct", direct["lightgbm_ranker"]),
                ("conditional", grid_models["lightgbm_ranker"]),
                ("basic_practice", basic["lightgbm_ranker"]),
            ):
                atomic_parquet_write(_importance(model), fold / f"importance_{name}.parquet")
            # Written last: interruption before this marker safely redoes this fold.
            atomic_json_write(fold / "complete.json", state)
        if not pd.Timestamp(state["training_end"]) < pd.Timestamp(state["prediction_date"]):
            raise ValueError("Cached fold violates chronology")
        for key, loss in state["losses"].items():
            histories.setdefault(key, []).append(loss)
        positions = np.asarray(state["q_positions"], dtype=float)
        if np.isfinite(positions).all() and (positions > 0).all():
            qhistory.append((np.asarray(state["q_scores"]), positions))
        dnf_history.extend(zip(state["raw_dnf"], state["actual_dnf"], strict=True))
        cutoffs.append(
            {
                "season": int(event["season"]),
                "round": int(event["round"]),
                "training_end": state["training_end"],
                "race_date": state["prediction_date"],
                "training_rows": state["training_rows"],
                "stage": state["stage"],
            }
        )
        if index >= warmup_races:
            outputs.append(predictions)
            qualifying_outputs.append(qforecast)
    _check_baseline(production, stage)
    predictions = pd.concat(outputs, ignore_index=True)
    qualifying = pd.concat(qualifying_outputs, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_parquet_write(predictions, output / "predictions.parquet")
    atomic_parquet_write(qualifying, output / "qualifying_predictions.parquet")
    atomic_text_write(output / "cutoffs.csv", pd.DataFrame(cutoffs).to_csv(index=False))
    final = cache / f"{int(folds.iloc[-1]['season'])}_{int(folds.iloc[-1]['round']):02d}"
    for name in ("qualifying", "direct", "conditional", "basic_practice"):
        atomic_text_write(
            output / f"importance_{name}.csv",
            pd.read_parquet(final / f"importance_{name}.parquet").to_csv(index=False),
        )
    return summarize_study(
        predictions, qualifying, output, params | {"run_id": run_id, "cache": str(cache)}
    )


def _race_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, season, round_number), group in predictions.groupby(
        ["candidate", "season", "round"]
    ):
        score = evaluate_predictions(group)
        winner = group.loc[group["target_win"].eq(1), "p_win"].iloc[0]
        rows.append(
            {
                "candidate": candidate,
                "season": season,
                "round": round_number,
                **{key: score[key] for key in ("spearman", "position_log_loss", "position_mae")},
                "winner_log_loss": -float(np.log(max(winner, 1e-12))),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(scores: pd.DataFrame, candidate: str, *, draws: int = 2000) -> dict[str, Any]:
    base = scores.loc[scores["candidate"].eq("direct_ensemble")].set_index(["season", "round"])
    other = scores.loc[scores["candidate"].eq(candidate)].set_index(["season", "round"])
    keys = base.index.intersection(other.index).sort_values()
    rng = np.random.default_rng(44)
    # Paired four-race moving blocks preserve some local dependence.
    starts = rng.integers(0, len(keys), size=(draws, int(np.ceil(len(keys) / 4))))
    indices = ((starts[..., None] + np.arange(4)) % len(keys)).reshape(draws, -1)[:, : len(keys)]
    result = {}
    for metric in ("spearman", "position_log_loss", "winner_log_loss", "position_mae"):
        delta = (other.loc[keys, metric] - base.loc[keys, metric]).to_numpy()
        estimates = delta[indices].mean(axis=1)
        result[metric] = {
            "candidate_minus_direct": float(delta.mean()),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        }
    return result


def summarize_study(
    predictions: pd.DataFrame,
    qualifying: pd.DataFrame,
    output: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    scores = _race_scores(predictions)
    metrics = {}
    for candidate, group in predictions.groupby("candidate", observed=True):
        values = evaluate_predictions(group)
        values["winner_log_loss"] = float(
            scores.loc[scores["candidate"].eq(candidate), "winner_log_loss"].mean()
        )
        metrics[candidate] = values
    qmetrics = []
    for (season, round_number), group in qualifying.groupby(["season", "round"]):
        actual = group["label_quali_position"]
        if actual.isna().any() or not actual.gt(0).all():
            continue
        actual_rank = actual.rank(method="first").to_numpy(dtype=int)
        size = len(group)
        matrix = group[[f"p_quali_position_{i}" for i in range(1, size + 1)]].to_numpy()
        matrix = 0.999 * matrix + 0.001 / size
        pole = actual_rank == 1
        qmetrics.append(
            {
                "season": season,
                "round": round_number,
                "spearman": float(
                    spearmanr(actual_rank, group["predicted_quali_position"]).statistic
                ),
                "position_mae": float(
                    np.mean(np.abs(actual_rank - group["predicted_quali_position"]))
                ),
                "pole_log_loss": -float(np.log(matrix[pole, 0][0])),
                "pole_accuracy": float(actual_rank[np.argmax(matrix[:, 0])] == 1),
                "position_log_loss": -float(
                    np.log(matrix[np.arange(size), actual_rank - 1]).mean()
                ),
            }
        )
    calibration = pd.concat(
        [calibration_table(g).assign(candidate=c) for c, g in predictions.groupby("candidate")],
        ignore_index=True,
    )
    atomic_text_write(output / "race_scores.csv", scores.to_csv(index=False))
    atomic_text_write(output / "qualifying_scores.csv", pd.DataFrame(qmetrics).to_csv(index=False))
    atomic_text_write(output / "calibration.csv", calibration.to_csv(index=False))
    years = {}
    for (candidate, season), group in predictions.groupby(["candidate", "season"]):
        years.setdefault(candidate, {})[str(season)] = evaluate_predictions(group)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": params,
        "production_unchanged": True,
        "promoted": False,
        "candidates": metrics,
        "by_season": years,
        "qualifying": {
            "races": len(qmetrics),
            **pd.DataFrame(qmetrics)
            .drop(columns=["season", "round"], errors="ignore")
            .mean()
            .to_dict(),
        },
        "paired_comparisons": {
            c: paired_bootstrap(scores, c) for c in metrics if c != "direct_ensemble"
        },
        "protocol": [
            "Identical held-out races, training support, practice recipe "
            "and calibrated pre-Q DNF probabilities.",
            "Retrain each race on strictly earlier race dates; "
            "earlier warmup races seed calibration.",
            "Qualifying and race temperatures use only earlier OOF forecasts; "
            "each fold updates state after freezing predictions.",
            "Conditional race training observes Q in earlier races only. "
            "Pre-Q inference marginalizes sampled full-grid permutations.",
            "No measured Q1/2/3 times, final starting grid, or current actual "
            "qualifying enter pre-Q scoring.",
            "No generated qualifying covariates are used in race-model training; "
            "no in-sample stacked predictions.",
            "Uniform 0.1% matrix smoothing preserves all grid probability constraints "
            "and avoids finite-MC zero losses.",
        ],
        "limitations": [
            "Retrospective, conditional on the archived race-entrant set; "
            "does not predict entry changes or DNQ.",
            "Qualifying order is not a penalty-adjusted starting grid; "
            "later penalties are not modeled.",
            "Historical publication/revision times are not reconstructed; "
            "2026 is reused retrospective validation.",
            "A smaller development tree profile is not proof of improvement "
            "in the full production configuration."
            if params["profile"] == "backtest"
            else "Full tree counts matched; this is still retrospective "
            "rather than prospective validation.",
            "Four-race block bootstrap is approximate and not adjusted "
            "for repeated research/model selection.",
        ],
    }
    atomic_json_write(output / "metrics.json", report)
    from f1predict.evaluation.qualifying_report import write_qualifying_report

    write_qualifying_report(report, calibration, qualifying, output)
    return report
