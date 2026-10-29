"""Forward-only driver/car ratings, layout residuals and separately typed reliability.

These are regularized predictive proxies, NOT an identifiable causal decomposition of
driver skill and car quality. Parameters are fixed before the research backtest.
Every entrant receives the same pre-event snapshot; only then may outcomes update state.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from f1predict.config import DataPaths
from f1predict.data.io import atomic_json_write, atomic_parquet_write, read_partitioned_table

KEYS = ["season", "round", "driver_id"]
RECIPE = "hierarchical_pairwise_ratings_v1"

# Sporting championship points are NEVER merged. Only latent model state is shared,
# conservatively discounted on brand changes. IDs outside these years remain separate.
LINEAGES = {
    "renault": (2010, 2020, "enstone"),
    "lotus_f1": (2012, 2015, "enstone"),
    "alpine": (2021, 2026, "enstone"),
    "force_india": (2010, 2018, "silverstone"),
    "racing_point": (2019, 2020, "silverstone"),
    "aston_martin": (2021, 2026, "silverstone"),
    "sauber": (2010, 2025, "hinwil"),
    "alfa": (2019, 2023, "hinwil"),
    "audi": (2026, 2026, "hinwil"),
    "toro_rosso": (2010, 2019, "faenza"),
    "alphatauri": (2020, 2023, "faenza"),
    "rb": (2024, 2026, "faenza"),
}

MECHANICAL_STATUSES = {
    "Engine",
    "Gearbox",
    "Brakes",
    "Hydraulics",
    "Suspension",
    "Power Unit",
    "Electrical",
    "Oil leak",
    "Mechanical",
    "Power loss",
    "Transmission",
    "Fuel pressure",
    "Exhaust",
    "Water pressure",
    "Water leak",
    "Turbo",
    "Overheating",
    "Battery",
    "Clutch",
    "ERS",
    "Electronics",
    "Throttle",
    "Driveshaft",
    "Steering",
    "Alternator",
    "Oil pressure",
    "Fuel pump",
    "Fuel system",
    "Technical",
    "Fuel leak",
    "Water pump",
    "Cooling system",
    "Differential",
    "Radiator",
    "Out of fuel",
    "Spark plugs",
    "Track rod",
    "Brake duct",
    "Heat shield fire",
    "Drivetrain",
    "Pneumatics",
    "Engine misfire",
}
INCIDENT_STATUSES = {"Collision", "Accident", "Collision damage", "Spun off"}


@dataclass(frozen=True)
class RatingConfig:
    driver_step: float = 0.035
    constructor_step: float = 0.25
    teammate_step: float = 0.10
    context_step: float = 0.20
    context_prior: float = 5.0
    driver_season_retention: float = 0.98
    constructor_season_retention: float = 0.70
    regulation_retention: float = 0.40
    rebrand_retention: float = 0.50
    reliability_retention_per_race: float = 0.985

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = RatingConfig()


def constructor_lineage(constructor: str, season: int) -> str:
    start, end, lineage = LINEAGES.get(constructor, (0, -1, constructor))
    return lineage if start <= season <= end else constructor


def regulation_era(season: int) -> str:
    for year, name in (
        (2026, "2026+"),
        (2022, "2022-2025"),
        (2017, "2017-2021"),
        (2014, "2014-2016"),
    ):
        if season >= year:
            return name
    return "2010-2013"


def load_event_layouts(archive_path: Path, races: pd.DataFrame) -> pd.DataFrame:
    """Use each race's archived layout, not today's circuit definition or actual distance.

    Course geometry is a pre-event input. Race distance/laps, flags, wetness, winners,
    championship-decider flags and later corrected publication times are not used.
    """
    with zipfile.ZipFile(archive_path) as archive:
        raw = pd.read_csv(archive.open("f1db-races.csv"), low_memory=False)
    raw = raw.rename(columns={"year": "season", "date": "layout_date"})
    fields = [
        "season",
        "round",
        "layout_date",
        "circuitLayoutId",
        "circuitType",
        "direction",
        "courseLength",
        "turns",
    ]
    result = (
        races[["season", "round", "race_date"]]
        .drop_duplicates()
        .merge(raw[fields], on=["season", "round"], how="left", validate="one_to_one")
    )
    matched = result["layout_date"].notna() & pd.to_datetime(
        result["layout_date"]
    ).dt.normalize().eq(pd.to_datetime(result["race_date"]).dt.normalize())
    exclusions = result.loc[~matched, ["season", "round", "race_date", "layout_date"]].to_dict(
        "records"
    )
    # Fail closed per event: schedule discrepancies do not justify guessing layout
    # identity, or borrowing the present-day circuit definition.
    for column in ("courseLength", "turns", "circuitType", "direction"):
        result.loc[~matched, column] = np.nan
    result["rating_layout_length_km"] = pd.to_numeric(result["courseLength"], errors="coerce")
    result["rating_layout_turns"] = pd.to_numeric(result["turns"], errors="coerce")
    valid = result["rating_layout_length_km"].gt(0) & result["rating_layout_turns"].gt(0)
    result.loc[~valid, ["rating_layout_length_km", "rating_layout_turns"]] = np.nan
    result["rating_layout_turns_per_km"] = (
        result["rating_layout_turns"] / result["rating_layout_length_km"]
    )
    result["rating_layout_type"] = result["circuitType"].fillna("UNKNOWN")
    result["rating_layout_direction"] = result["direction"].fillna("UNKNOWN")
    result["rating_layout_available"] = valid.astype("int8")
    density = pd.cut(
        result["rating_layout_turns_per_km"],
        [-np.inf, 2.5, 3.5, np.inf],
        labels=["low_turn_density", "medium_turn_density", "high_turn_density"],
    )
    length = pd.cut(
        result["rating_layout_length_km"],
        [-np.inf, 4.5, 6, np.inf],
        labels=["short", "medium", "long"],
    )
    result["layout_bucket"] = (
        result["rating_layout_type"].astype("string")
        + ":"
        + density.astype("string").fillna("unknown")
        + ":"
        + length.astype("string").fillna("unknown")
    )
    result["layout_key"] = result["circuitLayoutId"].fillna("UNKNOWN").astype(str)
    unknown = "unverified_" + result["season"].astype(str) + "_" + result["round"].astype(str)
    result.loc[~matched, "layout_key"] = unknown.loc[~matched]
    result.loc[~matched, "layout_bucket"] = unknown.loc[~matched]
    result = result[
        [
            "season",
            "round",
            "layout_key",
            "layout_bucket",
            *[c for c in result if c.startswith("rating_layout_")],
        ]
    ]
    result.attrs["layout_exclusions"] = exclusions
    return result


@dataclass
class Residual:
    value: float = 0.0
    observations: int = 0

    def estimate(self, prior: float) -> float:
        return self.value * self.observations / (self.observations + prior)


@dataclass
class PaceChannel:
    drivers: dict = field(default_factory=lambda: defaultdict(float))
    teams: dict = field(default_factory=lambda: defaultdict(float))
    driver_n: dict = field(default_factory=lambda: defaultdict(int))
    team_n: dict = field(default_factory=lambda: defaultdict(int))
    contexts: dict = field(default_factory=lambda: defaultdict(Residual))

    def update(self, event: pd.DataFrame, ranks: pd.Series, config: RatingConfig) -> None:
        selected = event.loc[ranks.notna() & ranks.gt(0)].copy()
        if len(selected) < 2:
            return
        drivers = selected["driver_id"].tolist()
        teams = selected["rating_lineage"].tolist()
        observed = ranks.loc[selected.index].to_numpy(dtype=float)
        scores = np.array(
            [self.drivers[d] + self.teams[t] for d, t in zip(drivers, teams, strict=True)]
        )
        actual = (observed[:, None] < observed[None, :]).astype(float)
        actual[observed[:, None] == observed[None, :]] = 0.5
        errors = actual - expit(scores[:, None] - scores[None, :])
        np.fill_diagonal(errors, 0)
        residual = errors.sum(axis=1) / (len(selected) - 1)
        # All gradients use the frozen pre-event state, never earlier row updates.
        d_updates, team_residuals = {}, defaultdict(list)
        for i, (driver, team) in enumerate(zip(drivers, teams, strict=True)):
            mates = [j for j, other in enumerate(teams) if other == team and j != i]
            mate_error = (
                np.mean(
                    [
                        actual[i, j] - expit(self.drivers[driver] - self.drivers[drivers[j]])
                        for j in mates
                    ]
                )
                if mates
                else 0.0
            )
            d_updates[driver] = config.driver_step * residual[i] + config.teammate_step * mate_error
            team_residuals[team].append(float(residual[i]))
        for driver, change in d_updates.items():
            self.drivers[driver] += change
            self.driver_n[driver] += 1
        for team, values in team_residuals.items():
            self.teams[team] += config.constructor_step * float(np.mean(values))
            self.team_n[team] += 1
        # Layout residuals do not feed back into global ratings: adding their feature
        # columns is an interpretable ablation over identical global rating states.
        for scope in ("track", "bucket"):
            context = str(selected["layout_key" if scope == "track" else "layout_bucket"].iloc[0])
            for driver, error in zip(drivers, residual, strict=True):
                cell = self.contexts[("driver", scope, driver, context)]
                cell.value += config.context_step * (float(error) - cell.value)
                cell.observations += 1
            for team, values in team_residuals.items():
                cell = self.contexts[("constructor", scope, team, context)]
                cell.value += config.context_step * (float(np.mean(values)) - cell.value)
                cell.observations += 1


@dataclass
class Reliability:
    starts: float = 0.0
    mechanical: float = 0.0
    incident: float = 0.0
    other: float = 0.0

    def features(self, prefix: str) -> dict:
        # Coherent Dirichlet prior [finish=9, mechanical=.4, incident=.4, other=.2].
        denominator = self.starts + 10
        return {
            f"{prefix}_dnf_rate": (1 + self.mechanical + self.incident + self.other) / denominator,
            f"{prefix}_mechanical_rate": (0.4 + self.mechanical) / denominator,
            f"{prefix}_incident_rate": (0.4 + self.incident) / denominator,
            f"{prefix}_other_rate": (0.2 + self.other) / denominator,
            f"{prefix}_evidence": self.starts,
        }


def _decay_reliability(state: Reliability, factor: float) -> None:
    for name in ("starts", "mechanical", "incident", "other"):
        setattr(state, name, getattr(state, name) * factor)


def build_rating_features(
    data: pd.DataFrame,
    layouts: pd.DataFrame,
    config: RatingConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict]:
    """Snapshot -> update after event. Supports historical and unlabelled live entrants."""
    required = [
        *KEYS,
        "race_date",
        "circuit_id",
        "constructor_id",
        "label_quali_position",
        "target_finish_position",
        "target_classified",
        "target_dnf",
        "target_dns",
        "target_dsq",
        "label_status",
    ]
    missing = set(required) - set(data)
    if missing or data.duplicated(KEYS).any():
        raise ValueError(f"Invalid rating input; missing={sorted(missing)} or duplicate entrants")
    frame = data[required].merge(
        layouts, on=["season", "round"], how="left", validate="many_to_one"
    )
    frame = frame.sort_values(["race_date", *KEYS], ignore_index=True)
    frame["rating_lineage"] = [
        constructor_lineage(str(t), int(y))
        for t, y in zip(frame.constructor_id, frame.season, strict=True)
    ]
    # Include circuit ID as protection when a source layout ID is missing/reused.
    frame["layout_key"] = (
        frame.circuit_id.astype(str) + ":" + frame.layout_key.fillna("UNKNOWN").astype(str)
    )
    channels = {"quali": PaceChannel(), "race": PaceChannel()}
    reliability = defaultdict(Reliability)
    prior_season, asof = None, pd.NaT
    last_brand, last_driver_team, team_rebrands, driver_moves = (
        {},
        {},
        defaultdict(int),
        defaultdict(int),
    )
    snapshots, events, update_counts = [], [], defaultdict(int)
    for (season, round_number, date), event in frame.groupby(
        ["season", "round", "race_date"], sort=False
    ):
        if event["layout_key"].nunique() != 1 or event["layout_bucket"].nunique() != 1:
            raise ValueError("One event has conflicting layout metadata")
        if prior_season is not None and season != prior_season:
            major = regulation_era(int(season)) != regulation_era(int(prior_season))
            team_retention = (
                config.regulation_retention if major else config.constructor_season_retention
            )
            for channel in channels.values():
                for d in channel.drivers:
                    channel.drivers[d] *= config.driver_season_retention
                for team in channel.teams:
                    channel.teams[team] *= team_retention
                for cell in channel.contexts.values():
                    cell.value *= team_retention
            for key, state in reliability.items():
                if key[0] in {"constructor", "track", "bucket"}:
                    _decay_reliability(state, team_retention)
        for state in reliability.values():
            _decay_reliability(state, config.reliability_retention_per_race)
        # A rebrand is known from the current entrants, before any performance label.
        teams = event[["rating_lineage", "constructor_id"]].drop_duplicates()
        if teams.rating_lineage.duplicated().any():
            raise ValueError("Two concurrent constructors map to one rating lineage")
        for team, brand in teams.itertuples(index=False, name=None):
            if team in last_brand and last_brand[team] != brand:
                team_rebrands[team] += 1
                for channel in channels.values():
                    channel.teams[team] *= config.rebrand_retention
                    for key, cell in channel.contexts.items():
                        if key[0] == "constructor" and key[2] == team:
                            cell.value *= config.rebrand_retention
                _decay_reliability(reliability[("constructor", team)], config.rebrand_retention)
                events.append(
                    {
                        "season": int(season),
                        "round": int(round_number),
                        "lineage": team,
                        "from": last_brand[team],
                        "to": brand,
                    }
                )
            last_brand[team] = brand
        for _, entrant in event.iterrows():
            d, t = entrant.driver_id, entrant.rating_lineage
            moved = d in last_driver_team and last_driver_team[d] != t
            driver_moves[d] += int(moved)
            row = {key: entrant[key] for key in KEYS}
            row.update(
                rating_lineage=t,
                rating_era=regulation_era(int(season)),
                rating_driver_new_team=int(moved),
                rating_driver_team_changes_pre=driver_moves[d],
                rating_constructor_rebrands_pre=team_rebrands[t],
                asof_race_date=asof,
            )
            last_driver_team[d] = t
            for label, channel in channels.items():
                prefix = f"rating_{label}"
                mates = event.loc[event.rating_lineage.eq(t) & event.driver_id.ne(d), "driver_id"]
                mate_rating = np.mean([channel.drivers[m] for m in mates]) if len(mates) else np.nan
                row.update(
                    {
                        f"{prefix}_driver": channel.drivers[d],
                        f"{prefix}_constructor": channel.teams[t],
                        f"{prefix}_combined": channel.drivers[d] + channel.teams[t],
                        f"{prefix}_teammate_gap": channel.drivers[d] - mate_rating,
                        f"{prefix}_driver_evidence": channel.driver_n[d],
                        f"{prefix}_constructor_evidence": channel.team_n[t],
                    }
                )
                for scope in ("track", "bucket"):
                    context = str(entrant["layout_key" if scope == "track" else "layout_bucket"])
                    for entity, key in (("driver", d), ("constructor", t)):
                        cell = channel.contexts[(entity, scope, key, context)]
                        row[f"rating_context_{label}_{entity}_{scope}"] = cell.estimate(
                            config.context_prior
                        )
                        row[f"rating_context_{label}_{entity}_{scope}_evidence"] = cell.observations
            for scope, key in (
                ("driver", d),
                ("constructor", t),
                ("track", entrant.layout_key),
                ("bucket", entrant.layout_bucket),
            ):
                row.update(reliability[(scope, key)].features(f"rel_{scope}"))
            row.update({c: entrant[c] for c in event if c.startswith("rating_layout_")})
            snapshots.append(row)
        # This boundary is the leakage contract: no rating is updated before all rows
        # above have been emitted. Qualifying is deliberately updated after the race too.
        channels["quali"].update(event, event.label_quali_position, config)
        finishers = (
            event.target_classified.eq(1)
            & event.target_dnf.eq(0)
            & event.target_dns.eq(0)
            & event.target_dsq.eq(0)
        )
        channels["race"].update(event, event.target_finish_position.where(finishers), config)
        update_counts["quali_observations"] += int(event.label_quali_position.gt(0).sum())
        update_counts["classified_pace_observations"] += int(finishers.sum())
        for _, entrant in event.iterrows():
            if pd.isna(entrant.target_dnf) or entrant.target_dns == 1 or entrant.target_dsq == 1:
                continue
            cause = None
            if entrant.target_dnf == 1:
                cause = (
                    "mechanical"
                    if entrant.label_status in MECHANICAL_STATUSES
                    else "incident"
                    if entrant.label_status in INCIDENT_STATUSES
                    else "other"
                )
                update_counts[f"reliability_{cause}"] += 1
            for scope, key in (
                ("driver", entrant.driver_id),
                ("constructor", entrant.rating_lineage),
                ("track", entrant.layout_key),
                ("bucket", entrant.layout_bucket),
            ):
                state = reliability[(scope, key)]
                state.starts += 1
                if cause:
                    setattr(state, cause, getattr(state, cause) + 1)
        if event.target_finish_position.notna().any():
            asof = pd.Timestamp(date)
        prior_season = season
    output = pd.DataFrame(snapshots)
    if not output.asof_race_date.isna().all():
        check = output.merge(data[[*KEYS, "race_date"]], on=KEYS, validate="one_to_one")
        if (pd.to_datetime(check.asof_race_date) >= pd.to_datetime(check.race_date)).any():
            raise AssertionError("Rating state was updated using the target/future race")
    return output, {
        "recipe": RECIPE,
        "config": config.to_dict(),
        "rebrands": events,
        "update_counts": dict(update_counts),
        "rows": len(output),
        "features": len([c for c in output if c.startswith(("rating_", "rel_"))]),
    }


def prepare_rating_features(paths: DataPaths, output: Path) -> tuple[pd.DataFrame, dict]:
    """Write a sidecar study table, leaving the established feature matrices untouched."""
    # Imported locally to keep the sequential state machine independent of study IO.
    from f1predict.evaluation.qualifying_study import load_study_data

    data, _ = load_study_data(paths)
    results = read_partitioned_table(paths.processed_jolpica, "results")
    data = data.merge(
        results[[*KEYS, "status"]].rename(columns={"status": "label_status"}),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    layouts = load_event_layouts(paths.raw_f1db / "f1db-csv.zip", data)
    table, manifest = build_rating_features(data, layouts)
    features = [c for c in table if c.startswith(("rating_", "rel_"))]
    context = [c for c in features if c.startswith(("rating_context_", "rating_layout_"))]
    reliability = [c for c in features if c.startswith("rel_")]
    manifest["families"] = {
        "global": [c for c in features if c not in context + reliability],
        "context": context,
        "reliability": reliability,
    }
    manifest["layout_races"] = len(layouts)
    manifest["layout_coverage"] = float(layouts.rating_layout_available.mean())
    manifest["layout_exclusions"] = layouts.attrs.get("layout_exclusions", [])
    manifest["policy"] = (
        "All entrant snapshots precede any event update. Qualifying and classified-only race "
        "pace channels update after the race. DNF cause histories are separate. Layout geometry "
        "is event-specific, not the current circuit description. Publication revisions unknown."
    )
    atomic_parquet_write(table, output / "ratings.parquet")
    atomic_json_write(output / "manifest.json", manifest)
    return table, manifest
