from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import date
from pathlib import Path
from typing import Annotated

import fastf1
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from f1predict import __version__
from f1predict.config import DataPaths, find_project_root
from f1predict.data.audit import build_audit
from f1predict.data.pipeline import sync_f1db, sync_fastf1, sync_jolpica, sync_openf1
from f1predict.features.audit import build_feature_audit
from f1predict.features.builder import build_feature_matrices

app = typer.Typer(
    name="f1predict",
    help="High-fidelity, leakage-safe Formula 1 outcome prediction.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Download, normalize, and audit source data.", no_args_is_help=True)
features_app = typer.Typer(
    help="Build and audit leakage-safe model features.", no_args_is_help=True
)
model_app = typer.Typer(help="Train and inspect probabilistic race models.", no_args_is_help=True)
evaluate_app = typer.Typer(help="Run and inspect walk-forward evaluation.", no_args_is_help=True)
stats_app = typer.Typer(help="Query historical driver and race tendencies.", no_args_is_help=True)
odds_app = typer.Typer(
    help="Compare model probabilities with bookmaker markets.", no_args_is_help=True
)
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(model_app, name="model")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(stats_app, name="stats")
app.add_typer(odds_app, name="odds")
console = Console()


def _paths(data_dir: Path | None) -> DataPaths:
    return DataPaths(data_dir.resolve()) if data_dir else DataPaths.default()


def _validate_range(start: int, end: int) -> None:
    if start < 1950:
        raise typer.BadParameter("start must be 1950 or later")
    if end < start:
        raise typer.BadParameter("end must be greater than or equal to start")
    if end > date.today().year + 1:
        raise typer.BadParameter("end is too far in the future")


@app.command()
def version() -> None:
    """Show the installed application version."""
    console.print(f"F1 Predict {__version__}")


@app.command()
def doctor(
    data_dir: Annotated[
        Path | None, typer.Option(help="Override the default project data directory.")
    ] = None,
) -> None:
    """Check the runtime, dependencies, paths, and local data catalog."""
    paths = _paths(data_dir)
    rows = [
        ("F1 Predict", __version__),
        ("Python", platform.python_version()),
        ("pandas", pd.__version__),
        ("FastF1", fastf1.__version__),
        (
            "FastF1 timing access",
            "credentials detected"
            if os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")
            else "not configured (Jolpica remains available)",
        ),
        ("Data directory", str(paths.root)),
        ("Catalog", "ready" if paths.catalog.exists() else "not created yet"),
    ]
    table = Table(title="System check", show_header=False)
    table.add_column("Item", style="cyan")
    table.add_column("Value")
    for row in rows:
        table.add_row(*row)
    console.print(table)


@data_app.command("sync")
def data_sync(
    start: Annotated[int, typer.Option(help="First season to download.")] = 2018,
    end: Annotated[int, typer.Option(help="Last season to download.")] = date.today().year,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore cached API responses and fetch again.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Sync compact schedules, results, qualifying, sprints, and pit stops from Jolpica."""
    _validate_range(start, end)
    paths = _paths(data_dir)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task(f"Syncing Jolpica seasons {start}–{end}…", total=None)
        summary = sync_jolpica(paths, start, end, refresh=refresh)
    table = Table(title="Jolpica sync complete")
    table.add_column("Season", justify="right")
    for name in ("races", "results", "qualifying", "sprints", "pit_stops"):
        table.add_column(name.replace("_", " ").title(), justify="right")
    for season, counts in summary.items():
        table.add_row(str(season), *(str(counts[name]) for name in counts))
    console.print(table)
    console.print(f"[dim]Data catalog: {paths.catalog}[/dim]")


@data_app.command("backfill")
def data_backfill(
    start: Annotated[int, typer.Option(help="First additional historical season.")] = 2010,
    end: Annotated[int, typer.Option(help="Last additional historical season.")] = 2017,
    staging_dir: Annotated[
        Path | None, typer.Option(help="Separate staging data root; production data stays intact.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Destination for the data coverage report.")
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option(help="Production baseline data directory.")
    ] = None,
) -> None:
    """Backfill older results and practice in a staging dataset, then audit coverage."""
    import httpx

    from f1predict.data.history import backfill_history

    _validate_range(start, end)
    root = find_project_root()
    stage = DataPaths((staging_dir or root / "work" / "history_expansion" / "data").resolve())
    output = (output_dir or root / "outputs" / "data_expansion").resolve()
    console.print(f"Staging older seasons in {stage.root}")

    def update(year: int, counts: dict[str, int]) -> None:
        console.print(
            f"{year}: {counts['races']} races, {counts['results']} driver results, "
            f"{counts['qualifying']} qualifying rows, {counts['pit_stops']} pit stops"
        )

    try:
        report = backfill_history(
            _paths(data_dir), stage, output, start=start, end=end, progress=update
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (OSError, httpx.HTTPError) as exc:
        console.print(f"Backfill download stopped: {exc}")
        console.print("Completed requests are cached. Rerun the same command to resume.")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Completed races: {report['baseline_races']} → {report['candidate_races']}; "
        f"driver rows: {report['baseline_driver_rows']} → {report['candidate_driver_rows']}."
    )
    console.print(
        f"Production unchanged: {report['production_unchanged']}. Report: {output / 'report.md'}"
    )


@data_app.command("sync-fastf1")
def data_sync_fastf1(
    start: Annotated[int, typer.Option(help="First season to download.")] = 2018,
    end: Annotated[int, typer.Option(help="Last season to download.")] = date.today().year,
    sessions: Annotated[
        str,
        typer.Option(help="Comma-separated session codes, e.g. FP1,FP2,FP3,Q,R."),
    ] = "FP1,FP2,FP3,Q,R",
    rounds: Annotated[
        str | None,
        typer.Option(help="Optional comma-separated round numbers; defaults to every round."),
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Rebuild existing session partitions.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Sync detailed FastF1 laps, tyres, session results, and weather."""
    _validate_range(start, end)
    paths = _paths(data_dir)
    requested = [value.strip().upper() for value in sessions.split(",") if value.strip()]
    allowed = {"FP1", "FP2", "FP3", "Q", "SQ", "S", "R"}
    invalid = sorted(set(requested) - allowed)
    if invalid:
        raise typer.BadParameter(f"unsupported session code(s): {', '.join(invalid)}")
    selected_rounds: set[int] | None = None
    if rounds:
        try:
            selected_rounds = {int(value.strip()) for value in rounds.split(",") if value.strip()}
        except ValueError as exc:
            raise typer.BadParameter("rounds must be comma-separated integers") from exc
        if not selected_rounds or min(selected_rounds) < 1:
            raise typer.BadParameter("rounds must contain positive integers")
    console.print(
        Panel.fit(
            "FastF1 detailed timing can take a while on the first run. "
            "Completed partitions are cached and skipped automatically.",
            title="Detailed data",
        )
    )
    outcomes = sync_fastf1(
        paths,
        start,
        end,
        requested,
        rounds=selected_rounds,
        refresh=refresh,
    )
    counts = pd.Series([row["status"] for row in outcomes]).value_counts().to_dict()
    console.print(
        f"[green]Downloaded {counts.get('downloaded', 0)}[/green], "
        f"cached {counts.get('cached', 0)}, "
        f"unavailable {counts.get('unavailable', 0)} session partitions."
    )
    unavailable = [row for row in outcomes if row["status"] == "unavailable"]
    if unavailable:
        console.print("[yellow]Unavailable sessions were skipped. First failures:[/yellow]")
        for row in unavailable[:5]:
            label = f"{row['season']} R{row['round']} {row['session']}"
            console.print(f"  {label}: {row.get('detail', 'unknown')}")


@data_app.command("sync-practice")
def data_sync_practice(
    start: Annotated[int, typer.Option(help="First season for F1DB classifications.")] = 2018,
    end: Annotated[int, typer.Option(help="Last season to download.")] = date.today().year,
    detailed_from: Annotated[
        int,
        typer.Option(help="First season for detailed OpenF1 laps, tyres, and weather."),
    ] = 2023,
    bulk: Annotated[
        bool, typer.Option(help="Use bounded season requests for a full backfill.")
    ] = False,
    rounds: Annotated[
        str | None,
        typer.Option(help="Optional OpenF1 round filter, e.g. 1,2. F1DB remains season-wide."),
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Redownload cached source data.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Sync practice classifications plus detailed laps, tyres, and weather."""
    _validate_range(start, end)
    paths = _paths(data_dir)
    selected_rounds: set[int] | None = None
    if rounds:
        try:
            selected_rounds = {int(value.strip()) for value in rounds.split(",") if value.strip()}
        except ValueError as exc:
            raise typer.BadParameter("rounds must be comma-separated integers") from exc
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Syncing F1DB practice classifications…", total=None)
        f1db_summary = sync_f1db(paths, start, end, refresh=refresh)
        progress.update(task, description="Syncing detailed OpenF1 practice sessions…")
        openf1_outcomes = sync_openf1(
            paths,
            max(2023, detailed_from),
            end,
            rounds=selected_rounds,
            refresh=refresh,
            bulk=bulk,
            progress=lambda row: progress.update(
                task,
                description=(
                    f"OpenF1 {row['season']} {row.get('session', row.get('endpoint', ''))}: "
                    f"{row['status']}"
                ),
            ),
        )
    counts = pd.Series([row["status"] for row in openf1_outcomes]).value_counts().to_dict()
    table = Table(title="Practice data sync complete")
    table.add_column("Source", style="cyan")
    table.add_column("Coverage")
    table.add_column("Rows / sessions", justify="right")
    table.add_row(
        "F1DB",
        f"{start}–{end} classifications",
        f"{sum(f1db_summary.values()):,} rows",
    )
    table.add_row(
        "OpenF1",
        f"{max(2023, detailed_from)}–{end} detailed",
        f"{counts.get('downloaded', 0)} new / {counts.get('cached', 0)} cached",
    )
    console.print(table)
    unavailable = [row for row in openf1_outcomes if row["status"] == "unavailable"]
    for row in unavailable[:5]:
        console.print(
            f"[yellow]{row['season']} R{row['round']} {row['session']}: "
            f"{row.get('detail', 'unavailable')}[/yellow]"
        )


@data_app.command("audit")
def data_audit(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON instead of tables.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Audit normalized data coverage, uniqueness, and joinability."""
    report = build_audit(_paths(data_dir))
    if json_output:
        console.print_json(json.dumps(report, default=str))
        return
    table = Table(title="Normalized data audit")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Seasons")
    table.add_column("Duplicates", justify="right")
    table.add_column("Status")
    for name, stats in report["tables"].items():
        season_range = (
            f"{stats.get('season_min')}–{stats.get('season_max')}"
            if stats.get("season_min") is not None
            else "—"
        )
        status = stats.get("status", "missing")
        color = "green" if status == "ok" else "yellow"
        table.add_row(
            name,
            str(stats.get("rows", 0)),
            season_range,
            str(stats.get("duplicate_keys", "—")),
            f"[{color}]{status}[/{color}]",
        )
    console.print(table)
    if report["coverage"]:
        console.print(
            Panel(json.dumps(report["coverage"], indent=2, default=str), title="Coverage")
        )
    f1db_rows = report["f1db"].get("practice_rows", 0)
    openf1_laps = report["openf1"].get("practice_laps", {}).get("rows", 0)
    openf1_sessions = report["openf1"].get("practice_laps", {}).get("sessions", 0)
    console.print(
        f"Practice detail: {f1db_rows:,} F1DB classifications; "
        f"{openf1_laps:,} OpenF1 lap rows across {openf1_sessions} sessions."
    )
    fastf1_rows = sum(item["rows"] for item in report["fastf1"].values())
    console.print(
        f"FastF1 detail: {fastf1_rows:,} rows across "
        f"{report['fastf1']['laps']['partitions']} session partition(s)."
    )
    for warning in report["warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


@features_app.command("build")
def features_build(
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Build pre- and post-qualifying race-driver feature matrices."""
    paths = _paths(data_dir)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Building leakage-safe feature matrices…", total=None)
        manifest = build_feature_matrices(paths)
    table = Table(title="Feature matrices ready")
    table.add_column("Mode", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Races", justify="right")
    table.add_column("Features", justify="right")
    for mode in ("pre_quali", "post_quali"):
        table.add_row(
            mode,
            f"{manifest['rows']:,}",
            f"{manifest['races']:,}",
            str(len(manifest["modes"][mode]["features"])),
        )
    console.print(table)
    console.print(f"[dim]Manifest: {paths.features / 'manifest.json'}[/dim]")


@features_app.command("audit")
def features_audit(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON instead of tables.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Verify feature coverage, uniqueness, and temporal leakage guards."""
    report = build_feature_audit(_paths(data_dir))
    if json_output:
        console.print_json(json.dumps(report, default=str))
        return
    table = Table(title="Feature audit")
    table.add_column("Mode", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Races", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Median coverage", justify="right")
    table.add_column("Status")
    for mode, stats in report["modes"].items():
        clean = not stats["duplicate_keys"] and not stats["all_null_features"]
        table.add_row(
            mode,
            f"{stats['rows']:,}",
            str(stats["races"]),
            str(stats["features"]),
            f"{stats['median_feature_coverage']:.1%}",
            "[green]ok[/green]" if clean else "[yellow]review[/yellow]",
        )
    console.print(table)
    coverage = report["coverage"]
    console.print(
        f"Post-quali coverage: qualifying position {coverage['post_quali_position']:.1%}; "
        f"grid position {coverage['post_grid_position']:.1%}."
    )
    console.print("[green]Leakage guards passed.[/green]")


@features_app.command("practice-checkpoint")
def features_practice_checkpoint(
    staging_dir: Annotated[Path, typer.Option(help="Isolated history/feature study data.")] = Path(
        "work/history_expansion/data"
    ),
    output: Annotated[Path, typer.Option(help="Audit report directory.")] = Path(
        "outputs/practice_features"
    ),
    data_dir: Annotated[Path | None, typer.Option(help="Frozen production baseline.")] = None,
) -> None:
    """Compare staged practice coverage and verify production was not changed."""
    from f1predict.features.study import practice_checkpoint

    report = practice_checkpoint(
        _paths(data_dir), DataPaths(staging_dir.resolve()), output.resolve()
    )
    console.print(
        f"[green]Practice checkpoint passed:[/green] {report['rows']:,} driver-race rows, "
        f"{report['races']} races. Production unchanged; no models trained."
    )
    console.print(f"Report: {output.resolve() / 'report.md'}")


@stats_app.command("driver-quali")
def stats_driver_quali(
    driver: Annotated[str, typer.Argument(help="Driver name, code, or ID.")],
    position: Annotated[int, typer.Argument(help="Qualifying position, e.g. 1 or 2.")],
    start_season: Annotated[
        int | None, typer.Option(help="Optional first season to include.")
    ] = None,
    end_season: Annotated[int | None, typer.Option(help="Optional last season to include.")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the summary and race history as JSON.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Show how a driver performs from one exact qualifying position."""
    from f1predict.analytics import driver_qualifying_position_history

    try:
        summary, history = driver_qualifying_position_history(
            _paths(data_dir),
            driver=driver,
            position=position,
            start_season=start_season,
            end_season=end_season,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        payload = {
            "summary": summary,
            "races": json.loads(history.to_json(orient="records", date_format="iso")),
        }
        console.print_json(json.dumps(payload, default=str))
        return

    overview = Table(title=f"{summary['driver_name']} when qualifying P{position}")
    overview.add_column("Starts", justify="right")
    overview.add_column("Wins", justify="right")
    overview.add_column("Podiums", justify="right")
    overview.add_column("Points", justify="right")
    overview.add_column("DNFs", justify="right")
    overview.add_column("Mean finish", justify="right")
    overview.add_column("Classified finish", justify="right")
    overview.add_column("Classified gained", justify="right")
    overview.add_row(
        str(summary["starts"]),
        f"{summary['wins']} ({summary['win_rate']:.1%})",
        f"{summary['podiums']} ({summary['podium_rate']:.1%})",
        f"{summary['points_finishes']} ({summary['points_rate']:.1%})",
        f"{summary['dnfs']} ({summary['dnf_rate']:.1%})",
        f"{summary['mean_finish']:.2f}",
        f"{summary['classified_mean_finish']:.2f}",
        f"{summary['classified_mean_positions_gained']:+.2f}",
    )
    console.print(overview)

    races = Table(title="Race history")
    races.add_column("Date")
    races.add_column("Race")
    races.add_column("Team")
    races.add_column("Finish", justify="right")
    races.add_column("Gained", justify="right")
    races.add_column("DNF", justify="center")
    for row in history.itertuples():
        races.add_row(
            str(pd.to_datetime(row.race_date).date()),
            str(row.race_name),
            str(row.constructor_name),
            str(int(row.target_finish_position)),
            f"{row.positions_gained:+.0f}",
            "yes" if row.target_dnf else "",
        )
    console.print(races)


@model_app.command("train")
def model_train(
    mode: Annotated[
        str,
        typer.Option(help="Model cutoff: pre_quali, post_quali, or all."),
    ] = "all",
    calibration_races: Annotated[
        int,
        typer.Option(help="Most recent races reserved for chronological calibration."),
    ] = 30,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Train LambdaRank ordering and calibrated reliability models."""
    from f1predict.models.training import train_model

    if mode not in {"pre_quali", "post_quali", "all"}:
        raise typer.BadParameter("mode must be pre_quali, post_quali, or all")
    if calibration_races < 10:
        raise typer.BadParameter("calibration-races must be at least 10")
    modes = ["pre_quali", "post_quali"] if mode == "all" else [mode]
    paths = _paths(data_dir)
    bundles = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Training models…", total=None)
        for selected in modes:
            progress.update(task, description=f"Training {selected} model…")
            bundles.append(train_model(paths, selected, calibration_races=calibration_races))
    table = Table(title="Model training complete")
    table.add_column("Mode", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Races", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Temperature", justify="right")
    table.add_column("DNF observed/calibrated", justify="right")
    for bundle in bundles:
        metadata = bundle.metadata
        dnf = metadata["dnf_calibration"]
        table.add_row(
            bundle.mode,
            f"{metadata['training_rows']:,}",
            str(metadata["training_races"]),
            str(metadata["features"]),
            f"{bundle.race_temperature:.2f}",
            f"{dnf['observed_rate']:.1%} / {dnf['calibrated_mean']:.1%}",
        )
    console.print(table)
    console.print(f"[dim]Artifacts: {paths.models}[/dim]")


@model_app.command("replay-deployment")
def replay_deployment(
    mode: Annotated[str, typer.Option(help="pre_quali or post_quali.")] = "post_quali",
    candidate: Annotated[str, typer.Option(help="Candidate name or selected.")] = "selected",
    driver: Annotated[str | None, typer.Option(help="Optional driver name or ID.")] = None,
    study_dir: Annotated[Path, typer.Option(help="Completed stage 5 report directory.")] = Path(
        "outputs/deployment_study"
    ),
    staging_dir: Annotated[Path, typer.Option(help="Isolated staging data.")] = Path(
        "work/history_expansion/data"
    ),
    simulations: Annotated[int | None, typer.Option(help="Override saved draw count.")] = None,
) -> None:
    """Replay the last studied race from its serialized pre-race model; never deploy."""
    from f1predict.evaluation.deployment_study import replay_deployment_candidate
    from f1predict.prediction import match_driver

    try:
        result = replay_deployment_candidate(
            DataPaths(staging_dir.resolve()),
            study_dir.resolve(),
            mode,
            candidate,
            simulations=simulations,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print("Research as-of replay; production model unchanged.")
    if driver:
        try:
            row = match_driver(result, driver)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        for table in _driver_tables(row, len(result)):
            console.print(table)
    else:
        console.print(_prediction_table(result))


@model_app.command("inspect")
def model_inspect(
    mode: Annotated[str, typer.Option(help="Model cutoff: pre_quali or post_quali.")] = (
        "post_quali"
    ),
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Show a trained model's provenance and calibration metadata."""
    from f1predict.models.training import load_model

    bundle = load_model(_paths(data_dir), mode)
    console.print(Panel(json.dumps(bundle.metadata, indent=2), title=f"{mode} model"))


@model_app.command("importance")
def model_importance(
    mode: Annotated[str, typer.Option(help="Model cutoff: pre_quali or post_quali.")] = (
        "post_quali"
    ),
    top: Annotated[int, typer.Option(help="Number of features to display.")] = 20,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Show gain-based feature importance from the ranking model."""
    from f1predict.models.training import load_model

    if top < 1:
        raise typer.BadParameter("top must be positive")
    importance = load_model(_paths(data_dir), mode).feature_importance().head(top)
    table = Table(title=f"Top {len(importance)} {mode} features")
    table.add_column("Rank", justify="right")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")
    for index, row in importance.iterrows():
        table.add_row(str(index + 1), str(row["feature"]), f"{row['importance']:.2%}")
    console.print(table)


@evaluate_app.command("qualifying-study")
def evaluate_qualifying_study(
    staging_dir: Annotated[Path, typer.Option(help="Isolated v2 feature dataset.")] = Path(
        "work/history_expansion/data"
    ),
    output: Annotated[Path, typer.Option(help="Study report and prediction directory.")] = Path(
        "outputs/qualifying_study"
    ),
    start_season: Annotated[int, typer.Option(help="First held-out season.")] = 2023,
    end_season: Annotated[int, typer.Option(help="Last held-out season.")] = 2026,
    profile: Annotated[
        str, typer.Option(help="Matched tree counts: backtest or full.")
    ] = "backtest",
    grid_scenarios: Annotated[int, typer.Option(help="Qualifying permutations per race.")] = 256,
    simulations: Annotated[
        int, typer.Option(help="Race draws; multiple of grid scenarios.")
    ] = 4096,
    warmup_races: Annotated[int, typer.Option(help="Earlier OOF calibration races.")] = 12,
    data_dir: Annotated[Path | None, typer.Option(help="Frozen production baseline.")] = None,
) -> None:
    """Backtest pre-Q forecasts that integrate uncertain qualifying grids; never promote."""
    from f1predict.evaluation.qualifying_study import run_qualifying_study

    _validate_range(start_season, end_season)
    report = run_qualifying_study(
        DataPaths(staging_dir.resolve()),
        _paths(data_dir),
        output.resolve(),
        start_season=start_season,
        end_season=end_season,
        profile=profile,
        grid_scenarios=grid_scenarios,
        simulations=simulations,
        warmup_races=warmup_races,
        progress=lambda message: console.print(message, markup=False),
    )
    table = Table(title="Qualifying-grid study (not promoted)")
    for name in ("Candidate", "Spearman", "Pos. log loss", "Winner log loss"):
        table.add_column(name)
    for name, metrics in report["candidates"].items():
        table.add_row(
            name,
            f"{metrics['spearman']:.4f}",
            f"{metrics['position_log_loss']:.4f}",
            f"{metrics['winner_log_loss']:.4f}",
        )
    console.print(table)
    console.print(f"Report: {output.resolve() / 'report.md'}")


@evaluate_app.command("rating-study")
def evaluate_rating_study(
    staging_dir: Annotated[Path, typer.Option(help="Isolated v2 data directory.")] = Path(
        "work/history_expansion/data"
    ),
    output: Annotated[Path, typer.Option(help="Rating study reports.")] = Path(
        "outputs/rating_study"
    ),
    start_season: Annotated[int, typer.Option(help="First held-out season.")] = 2023,
    end_season: Annotated[int, typer.Option(help="Last held-out season.")] = 2026,
    profile: Annotated[
        str, typer.Option(help="Matched tree profile: backtest or full.")
    ] = "backtest",
    simulations: Annotated[int, typer.Option(help="Joint race draws per forecast.")] = 4096,
    warmup_races: Annotated[int, typer.Option(help="Earlier OOF calibration races.")] = 12,
    data_dir: Annotated[Path | None, typer.Option(help="Frozen production baseline.")] = None,
) -> None:
    """Test driver/car ratings, layout effects and separate reliability; never promote."""
    from f1predict.evaluation.rating_study import run_rating_study

    _validate_range(start_season, end_season)
    report = run_rating_study(
        DataPaths(staging_dir.resolve()),
        _paths(data_dir),
        output.resolve(),
        start_season=start_season,
        end_season=end_season,
        profile=profile,
        simulations=simulations,
        warmup_races=warmup_races,
        progress=lambda text: console.print(text, markup=False),
    )
    table = Table(title="Rating study (not promoted)")
    for column in ("Candidate", "Spearman", "Position LL", "Winner LL", "DNF Brier"):
        table.add_column(column)
    for candidate, metrics in report["candidates"].items():
        table.add_row(
            candidate,
            f"{metrics['spearman']:.4f}",
            f"{metrics['position_log_loss']:.4f}",
            f"{metrics['winner_log_loss']:.4f}",
            f"{metrics['probabilities']['dnf']['brier']:.4f}",
        )
    console.print(table)
    console.print(f"Report: {output.resolve() / 'report.md'}")


@evaluate_app.command("deployment-study")
def evaluate_deployment_study(
    staging_dir: Annotated[Path, typer.Option(help="Isolated v2 data.")] = Path(
        "work/history_expansion/data"
    ),
    output: Annotated[Path, typer.Option(help="Stage 5 reports.")] = Path(
        "outputs/deployment_study"
    ),
    simulations: Annotated[int, typer.Option(help="Joint draws per forecast.")] = 4096,
    workers: Annotated[int, typer.Option(help="One-thread processes, 1–4.")] = 4,
    data_dir: Annotated[Path | None, typer.Option(help="Frozen production baseline.")] = None,
) -> None:
    """Run the fixed full-size pre/post-quali history/recency comparison; never promote."""
    from f1predict.evaluation.deployment_study import run_deployment_study

    try:
        report = run_deployment_study(
            DataPaths(staging_dir.resolve()),
            _paths(data_dir),
            output.resolve(),
            simulations=simulations,
            workers=workers,
            progress=lambda value: console.print(value, markup=False),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Full-size deployment study — not promoted")
    for column in ("Mode", "Development-selected", "2026 recommendation"):
        table.add_column(column)
    for mode, decision in report["decisions"].items():
        table.add_row(mode, decision["candidate"], decision["recommendation"])
    console.print(table)
    console.print(f"Report: {output.resolve() / 'report.md'}")


def _evaluation_table(summary: dict) -> Table:
    labels = {
        "pole_aware_ensemble": "Pole-aware ensemble",
        "anchored_ensemble": "Anchored ensemble",
        "random_forest": "Random Forest",
        "lightgbm_regression": "LightGBM regression",
        "lightgbm_ranker": "LightGBM LambdaRank",
    }
    table = Table(title="Walk-forward evaluation")
    table.add_column("Mode", style="cyan")
    table.add_column("Model")
    table.add_column("Spearman", justify="right")
    table.add_column("Top 3", justify="right")
    table.add_column("Top 10", justify="right")
    table.add_column("Pos. MAE", justify="right")
    table.add_column("Pos. log loss", justify="right")
    for mode, report in summary["modes"].items():
        for candidate, metrics in report["candidates"].items():
            table.add_row(
                mode,
                labels[candidate],
                f"{metrics['spearman']:.3f}",
                f"{metrics['top3_accuracy']:.1%}",
                f"{metrics['top10_accuracy']:.1%}",
                f"{metrics['position_mae']:.2f}",
                f"{metrics['position_log_loss']:.3f}",
            )
    return table


@evaluate_app.command("run")
def evaluate_run(
    mode: Annotated[
        str,
        typer.Option(help="Prediction cutoff: pre_quali, post_quali, or all."),
    ] = "all",
    start_season: Annotated[
        int,
        typer.Option(help="First season whose races are evaluated."),
    ] = 2023,
    simulations: Annotated[
        int,
        typer.Option(help="Monte Carlo samples per race and candidate."),
    ] = 5_000,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Run strict expanding-window backtests and build evaluation plots."""
    from f1predict.evaluation.backtest import run_walk_forward
    from f1predict.evaluation.report import build_report

    if mode not in {"pre_quali", "post_quali", "all"}:
        raise typer.BadParameter("mode must be pre_quali, post_quali, or all")
    if simulations < 500:
        raise typer.BadParameter("simulations must be at least 500")
    modes = ("pre_quali", "post_quali") if mode == "all" else (mode,)
    paths = _paths(data_dir)
    for selected in modes:
        with Progress(console=console) as progress:
            task = progress.add_task(f"Backtesting {selected}…", total=1)

            def update(
                done: int,
                total: int,
                race_name: str,
                task_id: int = task,
                selected_mode: str = selected,
            ) -> None:
                progress.update(
                    task_id,
                    total=total,
                    completed=done,
                    description=f"{selected_mode}: {race_name}",
                )

            run_walk_forward(
                paths,
                selected,
                start_season=start_season,
                simulations=simulations,
                progress=update,
            )
    summary = build_report(paths, modes)
    console.print(_evaluation_table(summary))
    console.print(f"[dim]Report: {paths.evaluation / 'report.md'}[/dim]")


@evaluate_app.command("report")
def evaluate_report(
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Display the latest saved walk-forward results."""
    paths = _paths(data_dir)
    path = paths.evaluation / "summary.json"
    if not path.exists():
        raise RuntimeError("No evaluation report found; run `f1predict evaluate run` first.")
    console.print(_evaluation_table(json.loads(path.read_text(encoding="utf-8"))))


@evaluate_app.command("audit")
def evaluate_audit(
    mode: Annotated[
        str,
        typer.Option(help="Prediction cutoff: pre_quali, post_quali, or all."),
    ] = "all",
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Audit chronology and joint probability consistency."""
    from f1predict.evaluation.audit import audit_backtest

    if mode not in {"pre_quali", "post_quali", "all"}:
        raise typer.BadParameter("mode must be pre_quali, post_quali, or all")
    modes = ("pre_quali", "post_quali") if mode == "all" else (mode,)
    table = Table(title="Backtest integrity audit")
    table.add_column("Mode", style="cyan")
    table.add_column("Races", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Chronology violations", justify="right")
    table.add_column("Max probability error", justify="right")
    table.add_column("Status")
    for selected in modes:
        report = audit_backtest(_paths(data_dir), selected)
        maximum_error = max(
            report["driver_probability_max_error"],
            report["grid_win_probability_max_error"],
            report["position_occupancy_max_error"],
        )
        table.add_row(
            selected,
            str(report["races"]),
            f"{report['rows']:,}",
            str(report["chronology_violations"]),
            f"{maximum_error:.2e}",
            "[green]passed[/green]" if report["passed"] else "[red]failed[/red]",
        )
    console.print(table)


@odds_app.command("compare")
def odds_compare(
    prediction_csv: Annotated[Path, typer.Argument(help="Saved f1predict prediction CSV.")],
    odds_csv: Annotated[
        Path,
        typer.Argument(
            help="Complete winner market with driver_id and decimal, American, or fractional odds."
        ),
    ],
    winner: Annotated[
        str | None,
        typer.Option(help="Optional actual winner name or ID for log-loss and Brier scoring."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional path for the joined comparison CSV."),
    ] = None,
) -> None:
    """De-vig a complete race-winner market and compare it with model P(win)."""
    from f1predict.bookmakers import compare_winner_market, load_odds, score_winner_market

    try:
        predictions = pd.read_csv(prediction_csv)
        odds = load_odds(odds_csv)
        comparison = compare_winner_market(predictions, odds)
        score = score_winner_market(comparison, winner) if winner else None
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Model vs normalized winner market")
    table.add_column("Driver")
    table.add_column("Model", justify="right")
    table.add_column("Market", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Book odds", justify="right")
    table.add_column("Model fair", justify="right")
    for row in comparison.itertuples():
        color = "green" if row.probability_delta > 0 else "red"
        table.add_row(
            str(row.driver_name),
            f"{row.p_win:.1%}",
            f"{row.market_probability:.1%}",
            f"[{color}]{row.probability_delta:+.1%}[/{color}]",
            f"{row.decimal_odds:.2f}",
            f"{row.model_fair_odds:.2f}" if math.isfinite(row.model_fair_odds) else "∞",
        )
    console.print(table)
    console.print(
        f"Raw implied-probability sum: {odds.attrs['overround']:.1%}. "
        "Comparison normalizes across the supplied driver field."
    )
    if score:
        scoring = Table(title=f"Winner scoring · {score['winner']}")
        scoring.add_column("Forecast")
        scoring.add_column("Log loss", justify="right")
        scoring.add_column("Brier", justify="right")
        scoring.add_row("Model", f"{score['model_log_loss']:.3f}", f"{score['model_brier']:.3f}")
        scoring.add_row(
            "De-vigged market", f"{score['market_log_loss']:.3f}", f"{score['market_brier']:.3f}"
        )
        console.print(scoring)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output, index=False)
        console.print(f"[dim]Comparison CSV: {output.resolve()}[/dim]")
    console.print("[dim]Forecast comparison only—not betting advice. Gamble responsibly.[/dim]")


@odds_app.command("history")
def odds_history(
    manifest_csv: Annotated[
        Path,
        typer.Argument(help="CSV with race, prediction_csv, odds_csv, and winner columns."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Optional path for the per-race scoring CSV."),
    ] = None,
    include_unverified: Annotated[
        bool,
        typer.Option("--include-unverified", help="Show an explicitly exploratory aggregate."),
    ] = False,
) -> None:
    """Score model and bookmaker winner probabilities over multiple races."""
    from f1predict.bookmakers import score_history, summarize_history

    try:
        scores = score_history(manifest_csv)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = summarize_history(scores, include_unverified=include_unverified)

    table = Table(title="Model vs market history")
    table.add_column("Race")
    table.add_column("Winner")
    table.add_column("Model favorite")
    table.add_column("Market favorite")
    table.add_column("Model LL", justify="right")
    table.add_column("Market LL", justify="right")
    table.add_column("Timing")
    for row in scores.itertuples():
        table.add_row(
            str(row.race),
            str(row.winner),
            str(row.model_favorite),
            str(row.market_favorite),
            f"{row.model_log_loss:.3f}",
            f"{row.market_log_loss:.3f}",
            "aligned" if row.timing_eligible else "unverified",
        )
    console.print(table)

    label = "Exploratory aggregate" if include_unverified else "Timing-aligned aggregate"
    totals = Table(title=f"{label} · {summary['races']} races")
    totals.add_column("Forecast")
    totals.add_column("Favorite accuracy", justify="right")
    totals.add_column("Mean log loss", justify="right")
    totals.add_column("Mean Brier", justify="right")
    if summary["races"]:
        for key, name in (("model", "Model"), ("market", "Normalized market")):
            totals.add_row(
                name,
                f"{summary[f'{key}_favorite_accuracy']:.1%}",
                f"{summary[f'{key}_mean_log_loss']:.3f}",
                f"{summary[f'{key}_mean_brier']:.3f}",
            )
        console.print(totals)
    else:
        console.print(
            "No races have verified timing metadata. "
            "Use --include-unverified to inspect an exploratory aggregate."
        )
    for row in scores.loc[~scores["timing_eligible"]].itertuples():
        console.print(f"[dim]{row.race}: {row.alignment_note}[/dim]")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(output, index=False)
        console.print(f"[dim]History CSV: {output.resolve()}[/dim]")
    console.print("[dim]Forecast comparison only—not betting advice. Gamble responsibly.[/dim]")


@odds_app.command("import-betfair")
def odds_import_betfair(
    stream: Annotated[Path, typer.Argument(help="Extracted Betfair NDJSON or .bz2 stream.")],
    driver_map: Annotated[
        Path, typer.Argument(help="JSON object mapping selection IDs to canonical driver IDs.")
    ],
    market_id: Annotated[str, typer.Option(help="Explicit race-winner market ID, e.g. 1.12345.")],
    market_type: Annotated[
        str, typer.Option(help="Exact source winner code: WIN, WINNER, or OUTRIGHT_WINNER.")
    ],
    cutoff: Annotated[str, typer.Option(help="Exact comparison cutoff with UTC offset.")],
    output: Annotated[Path, typer.Option(help="Destination odds CSV; metadata saved beside it.")],
    max_age_seconds: Annotated[
        int, typer.Option(help="Maximum runner price publication age at cutoff.")
    ] = 300,
) -> None:
    """Import timestamped Betfair archive prices using only messages before a cutoff."""
    from f1predict.data.betfair import import_betfair_snapshot
    from f1predict.data.io import atomic_json_write

    try:
        mapping = json.loads(driver_map.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("Driver map must be a JSON object")
        quotes, metadata = import_betfair_snapshot(
            stream,
            mapping,
            market_id,
            cutoff,
            max_age_seconds=max_age_seconds,
            expected_market_type=market_type,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(output, index=False)
    atomic_json_write(output.with_suffix(".metadata.json"), metadata)
    console.print(f"Imported {len(quotes)} drivers at {metadata['cutoff']}: {output.resolve()}")
    console.print(str(metadata["caveat"]))


@odds_app.command("fetch-polymarket")
def odds_fetch_polymarket(
    event_slug: Annotated[str, typer.Argument(help="Exact Polymarket race-winner event slug.")],
    driver_map: Annotated[
        Path, typer.Argument(help="JSON object mapping canonical driver IDs to exact market slugs.")
    ],
    prediction_csv: Annotated[
        Path, typer.Option(help="Model CSV specifying the full actual grid.")
    ],
    cutoff: Annotated[str, typer.Option(help="Exact pre-race comparison cutoff with UTC offset.")],
    output: Annotated[Path, typer.Option(help="Destination quotes CSV and provenance sidecars.")],
    max_age_seconds: Annotated[int, typer.Option(help="Maximum recorded price age.")] = 300,
) -> None:
    """Download a public historical prediction-market snapshot with timestamps."""
    import httpx

    from f1predict.data.io import atomic_json_write
    from f1predict.data.polymarket import PolymarketClient

    try:
        mapping = json.loads(driver_map.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("Driver map must be a JSON object")
        predictions = pd.read_csv(prediction_csv)
        if "driver_id" not in predictions:
            raise ValueError("Prediction CSV needs driver_id")
        with PolymarketClient() as client:
            snapshot = client.fetch_snapshot(
                event_slug,
                driver_markets=mapping,
                expected_driver_ids=predictions["driver_id"].tolist(),
                cutoff=cutoff,
                max_age_seconds=max_age_seconds,
            )
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.quotes.to_csv(output, index=False)
    raw_path = output.with_suffix(".raw.json")
    atomic_json_write(raw_path, snapshot.raw)
    snapshot.metadata["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    snapshot.metadata["prediction_sha256"] = hashlib.sha256(prediction_csv.read_bytes()).hexdigest()
    atomic_json_write(output.with_suffix(".metadata.json"), snapshot.metadata)
    console.print(f"Saved {len(snapshot.quotes)} driver histories at {cutoff}: {output.resolve()}")
    console.print(
        f"Unmapped observed mass: {snapshot.metadata['unmatched_observed_probability_sum']:.2%}; "
        f"unmapped histories missing: {snapshot.metadata['unmatched_missing_histories']}."
    )
    console.print("Review unmatched outcomes in the metadata before scoring a normalized field.")


def _prediction_table(predictions: pd.DataFrame) -> Table:
    table = Table(title="Predicted finishing order")
    table.add_column("Pos", justify="right", style="bold cyan")
    table.add_column("Driver")
    table.add_column("Team", style="dim")
    table.add_column("Expected", justify="right")
    table.add_column("Win", justify="right")
    table.add_column("Podium", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("DNF", justify="right")
    for _, row in predictions.iterrows():
        table.add_row(
            str(int(row["predicted_position"])),
            str(row["driver_name"]),
            str(row["constructor_name"]),
            f"{row['expected_position']:.2f}",
            f"{row['p_win']:.1%}",
            f"{row['p_podium']:.1%}",
            f"{row['p_points']:.1%}",
            f"{row['p_dnf']:.1%}",
        )
    return table


def _driver_tables(row: pd.Series, grid_size: int) -> tuple[Table, Table]:
    summary = Table(title=str(row["driver_name"]), show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    for label, value in (
        ("Predicted finish", f"P{int(row['predicted_position'])}"),
        ("Expected position", f"{row['expected_position']:.2f}"),
        ("Win", f"{row['p_win']:.2%}"),
        ("Podium", f"{row['p_podium']:.2%}"),
        ("Points", f"{row['p_points']:.2%}"),
        ("DNF", f"{row['p_dnf']:.2%}"),
    ):
        summary.add_row(label, value)
    distribution = Table(title="Full finishing-position distribution")
    distribution.add_column("Position", justify="right", style="cyan")
    distribution.add_column("Probability", justify="right")
    for position in range(1, grid_size + 1):
        distribution.add_row(str(position), f"{row[f'p_position_{position}']:.2%}")
    return summary, distribution


@app.command("predict")
def predict(
    race: Annotated[
        str,
        typer.Argument(help="Round number or race/circuit/location name, e.g. 19 or monza."),
    ],
    season: Annotated[int, typer.Option(help="Championship season.")] = date.today().year,
    mode: Annotated[
        str,
        typer.Option(help="auto, pre_quali, or post_quali."),
    ] = "auto",
    driver: Annotated[
        str | None,
        typer.Option(help="Optional driver name, code, or ID for a single-driver view."),
    ] = None,
    simulations: Annotated[
        int,
        typer.Option(help="Number of joint Monte Carlo race simulations."),
    ] = 20_000,
    seed: Annotated[int, typer.Option(help="Reproducible simulation seed.")] = 44,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the complete prediction as JSON."),
    ] = False,
    save: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Save CSV and JSON prediction files."),
    ] = True,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """Predict a full race or answer a single-driver probability query."""
    from f1predict.models.selection import write_production_selection
    from f1predict.prediction import match_driver, predict_race, prediction_payload

    if simulations < 500:
        raise typer.BadParameter("simulations must be at least 500")
    paths = _paths(data_dir)
    write_production_selection(paths)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Building race forecast…", total=None)
        try:
            result = predict_race(
                paths,
                season=season,
                race=race,
                mode=mode,
                simulations=simulations,
                seed=seed,
                save=save,
            )
        except (ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    if json_output:
        console.print_json(json.dumps(prediction_payload(result), indent=2, default=str))
        return
    console.print(
        Panel.fit(
            f"[bold]{result.event.season} {result.event.race_name}[/bold]\n"
            f"Mode: {result.mode} · {result.kind.replace('_', ' ')}\n"
            f"Roster: {result.roster_source}\n"
            f"Model: {result.metadata['model']} · {simulations:,} simulations",
            title="F1 Predict",
        )
    )
    if result.kind == "historical_replay":
        console.print(
            "[yellow]Historical replay uses today's fully trained model; use the evaluation "
            "artifacts for unbiased historical scores.[/yellow]"
        )
    if driver:
        try:
            row = match_driver(result.predictions, driver)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        summary, distribution = _driver_tables(row, len(result.predictions))
        console.print(summary)
        console.print(distribution)
    else:
        console.print(_prediction_table(result.predictions))
    if result.csv_path and result.json_path:
        console.print(f"[dim]CSV: {result.csv_path}[/dim]")
        console.print(f"[dim]JSON: {result.json_path}[/dim]")


if __name__ == "__main__":
    app()
