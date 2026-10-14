import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from data.collectors.backfill import ArchiveDownloader, backfill_symbol
from data.collectors.binance_rest import BinanceRest, RateLimitedError, parse_symbol_info
from data.collectors.live import LiveCollector
from data.config import get_settings
from data.quality.checks import (
    STEP, DataQualityError, assert_trainable, run_quality_checks,
)
from data.storage.db import connect, run_migrations
from data.storage.repository import last_candle_time, refresh_aggregates, upsert_symbol
from features.build import BuildResult, build_symbol
from features.pipeline import FEATURE_TIMEFRAMES
from features.summary import market_state
from ml import store as ml_store
from ml.dataset import build_dataset
from ml.artifacts import PREDICT_MODELS, ArtifactMismatch, save_model
from ml.diagnose import render_diagnosis, split_skill
from ml.evaluate import evaluate as ml_evaluate
from ml.folds import DEFAULT_FOLDS, final_split, holdout_fold, walk_forward_folds
from ml.models import make_models
from ml.predict import NoRecentFeatures, prediction_state
from ml.predict import render as render_prediction
from ml.labels import HORIZONS
from ml.targets import TARGETS, get_target
from ml.load import StaleFeaturesError, check_fresh, load_symbol_data, run_config
from ml.report import render_report

app = typer.Typer(help="AI Trading Buddy data foundation")
db_app = typer.Typer(help="Database maintenance")
symbols_app = typer.Typer(help="Symbol metadata")
features_app = typer.Typer(help="Market feature engine")
app.add_typer(db_app, name="db")
app.add_typer(symbols_app, name="symbols")
app.add_typer(features_app, name="features")
ml_app = typer.Typer(help="Prediction models (measurements, not advice)")
app.add_typer(ml_app, name="ml")
backtest_app = typer.Typer(help="Simulations on stored history (not advice)")
app.add_typer(backtest_app, name="backtest")
risk_app = typer.Typer(help="Your risk settings, used by tb signals")
app.add_typer(risk_app, name="risk")
bots_app = typer.Typer(help="Your Binance grid bots vs holding the coin "
                       "(typed in; not advice)")
app.add_typer(bots_app, name="bots")
console = Console()

# Walk-forward fold settings; a module attribute so tests can shrink them.
ML_FOLDS = DEFAULT_FOLDS

# A 1m feed more than this far behind is stale, not merely quiet.
STALE_AFTER_MINUTES = 5


def _setup_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_from_date(value: str) -> datetime:
    """Parse a `--from YYYY-MM-DD` CLI value as a UTC midnight datetime."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise typer.BadParameter(f"expected YYYY-MM-DD, got {value!r}")


@db_app.command("upgrade")
def db_upgrade():
    """Apply database migrations."""
    _setup_logging()
    applied = run_migrations()
    console.print(f"applied: {applied or 'nothing new'}")


@db_app.command("refresh-aggregates")
def db_refresh_aggregates(
    from_: str = typer.Option(
        None, "--from", help="Start date YYYY-MM-DD (UTC); default is the "
        "earliest stored 1m candle"
    ),
):
    """Materialize 5m/15m/1h/4h/1d over the stored 1m history.

    The refresh policies only reach back 3-365 days. `tb backfill` refreshes
    what it writes; use this for history loaded before that existed, or
    after an interrupted backfill.
    """
    start = _parse_from_date(from_) if from_ else None
    _setup_logging()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT min(open_time), max(open_time) FROM candles_1m")
            first, last = cur.fetchone()
        conn.commit()  # end the read transaction before the refresh
        if first is None:
            console.print("no 1m candles stored; nothing to refresh")
            return
        start = max(start, first) if start else first
        views = refresh_aggregates(conn, start, last + timedelta(minutes=1))
    console.print(f"refreshed {', '.join(views) or 'nothing'} over "
                  f"{start:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC")


@symbols_app.command("sync")
def symbols_sync():
    """Refresh symbol metadata and listing dates from Binance."""
    _setup_logging()
    settings = get_settings()
    api = BinanceRest()
    info = parse_symbol_info(api.exchange_info(settings.symbols))
    with connect() as conn:
        for symbol, fields in info.items():
            fields["listed_at"] = api.first_candle_time(symbol)
            upsert_symbol(conn, **fields)
            console.print(f"{symbol}: listed {fields['listed_at']:%Y-%m-%d}")


@app.command()
def backfill(
    symbol: str = typer.Option(None, help="One symbol; default is all configured"),
    from_: str = typer.Option(
        None, "--from", help="Start date YYYY-MM-DD (UTC); default is the "
        "symbol's own listing date"
    ),
):
    """Load historical 1m candles. Resumable: safe to re-run."""
    start = _parse_from_date(from_) if from_ else None
    _setup_logging()
    settings = get_settings()
    targets = [symbol.upper()] if symbol else settings.symbols
    api = BinanceRest()
    downloader = ArchiveDownloader(settings.binance_data_url)
    failures = []
    with connect() as conn:
        for i, s in enumerate(targets):
            try:
                written = backfill_symbol(conn, s, downloader, api, start=start)
                console.print(f"{s}: {written:,} candles written")
            except RateLimitedError as exc:
                # A 429/418 means Binance itself is telling us to stop.
                # Isolating this like an ordinary per-symbol failure would
                # let the loop immediately call Binance again for the next
                # symbol (e.g. first_candle_time), and requests made while
                # banned can extend the ban. Abort the whole run instead;
                # backfill is resumable, so re-running after retry_after
                # has elapsed picks up exactly where this left off.
                remaining = targets[i + 1:]
                logging.getLogger(__name__).error(
                    "%s: rate limited (HTTP %s), retry after %.0fs; "
                    "aborting backfill; not yet processed: %s",
                    s, exc.status, exc.retry_after, ", ".join(remaining) or "none",
                )
                console.print(
                    f"[red]{s}: rate limited (HTTP {exc.status}); "
                    f"retry after {exc.retry_after:.0f}s[/red]"
                )
                if remaining:
                    console.print(
                        f"[red]not yet processed: {', '.join(remaining)}[/red]"
                    )
                raise typer.Exit(code=1)
            except Exception as exc:
                # One symbol's failure must not stop the others; the failure
                # is still surfaced via logging, console output, and a
                # non-zero exit code once every symbol has been attempted.
                # Roll back first: an unguarded DB error inside
                # backfill_symbol leaves this shared, non-autocommit
                # connection's transaction aborted. Without a rollback, the
                # next symbol's very first query on this same connection
                # would raise InFailedSqlTransaction and be falsely reported
                # as failed too, even though nothing is wrong with it.
                try:
                    conn.rollback()
                except Exception:
                    pass  # a broken connection must not mask the original error
                logging.getLogger(__name__).error(
                    "%s: backfill failed: %s", s, exc
                )
                console.print(f"[red]{s}: backfill failed: {exc}[/red]")
                failures.append(s)
    if failures:
        console.print(f"[red]failed: {', '.join(failures)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def live():
    """Run the live collector until interrupted."""
    _setup_logging()
    collector = LiveCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        console.print("stopped")


@app.command()
def quality(
    symbol: str = typer.Option(None),
    timeframe: str = typer.Option("1m"),
):
    """Run data-quality checks and print the report."""
    # Validate arguments before touching configuration, so a bad argument
    # reports itself rather than a confusing config error.
    if timeframe not in STEP:
        raise typer.BadParameter(f"unknown timeframe {timeframe!r}")
    _setup_logging()
    settings = get_settings()
    targets = [symbol.upper()] if symbol else settings.symbols
    failed = []
    with connect() as conn:
        for s in targets:
            report = run_quality_checks(conn, s, timeframe)
            console.print(report.render())
            if report.verdict == "FAIL":
                failed.append(s)
    if failed:
        # Scripts and schedulers see a FAIL through the exit code.
        console.print(f"[red]quality FAIL: {', '.join(failed)}[/red]")
        raise typer.Exit(code=1)


def _result_line(r: BuildResult) -> str:
    label = f"{r.symbol} {r.timeframe}"
    if r.skipped_reason:
        return f"[red]{label}: skipped ({escape(r.skipped_reason)})[/red]"
    if r.rows_written == 0:
        return f"{label}: up to date ({r.seconds:.1f}s)"
    return (f"{label}: {r.rows_written:,} rows, {r.start:%Y-%m-%d %H:%M} .. "
            f"{r.end:%Y-%m-%d %H:%M} UTC, {'full' if r.full else 'incremental'}, "
            f"{r.seconds:.1f}s")


def _build_symbols(conn, targets: list[str], timeframes: tuple[str, ...],
                   rebuild: bool, problems: dict[str, str] | None = None) -> bool:
    """Build each symbol, print one line per (symbol, timeframe), and return
    True if anything was skipped or failed. When `problems` is given, each
    symbol that failed gets its first problem there ("data quality FAIL",
    "error: ..." when the build raised, or "<timeframe>: <reason>").

    `build_symbol` isolates the timeframes. This also isolates the symbols,
    so an error in the quality gate itself is reported and the rest go on.
    """
    failed = False
    for s in targets:
        try:
            results = build_symbol(conn, s, timeframes, rebuild=rebuild)
        except Exception as exc:
            # Same reasoning as `backfill`: an aborted transaction on this
            # shared connection would fail the next symbol's first query.
            try:
                conn.rollback()
            except Exception:
                pass  # a broken connection must not mask the original error
            logging.getLogger(__name__).error(
                "%s: feature build failed: %s", s, exc)
            console.print(f"[red]{s}: feature build failed: "
                          f"{escape(str(exc))}[/red]", soft_wrap=True)
            failed = True
            if problems is not None:
                problems.setdefault(s, f"error: {exc}")
            continue
        for r in results:
            console.print(_result_line(r), soft_wrap=True)
            failed = failed or r.skipped_reason is not None
            if problems is not None and r.skipped_reason is not None:
                problems.setdefault(s, r.skipped_reason if r.skipped_reason ==
                                    "data quality FAIL" else
                                    f"{r.timeframe}: {r.skipped_reason}")
    return failed


@features_app.command("build")
def features_build(
    symbol: str = typer.Option(None, help="One symbol; default is all configured"),
    timeframe: str = typer.Option(
        None, help="One of " + ", ".join(FEATURE_TIMEFRAMES) + "; default is all"),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Recompute everything from full history"),
):
    """Build market features from the stored candles. Safe to re-run."""
    if timeframe is not None and timeframe not in FEATURE_TIMEFRAMES:
        raise typer.BadParameter(
            f"unknown timeframe {timeframe!r}; "
            f"expected one of {', '.join(FEATURE_TIMEFRAMES)}")
    _setup_logging()
    targets = [symbol.upper()] if symbol else get_settings().symbols
    timeframes = (timeframe,) if timeframe else FEATURE_TIMEFRAMES
    with connect() as conn:
        failed = _build_symbols(conn, targets, timeframes, rebuild)
    if failed:
        console.print("[red]some builds failed or were skipped[/red]")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="For example BTCUSDT"),
    no_build: bool = typer.Option(
        False, "--no-build", help="Show the stored features without building"),
):
    """Print the current market state of one symbol on every timeframe."""
    _setup_logging()
    symbol = symbol.upper()
    with connect() as conn:
        if last_candle_time(conn, symbol) is None:
            console.print(f"[red]{escape(symbol)}: no 1m candles stored; "
                          "run tb backfill first[/red]", soft_wrap=True)
            raise typer.Exit(code=1)
        failed = False
        if not no_build:
            failed = _build_symbols(conn, [symbol], FEATURE_TIMEFRAMES, False)
        state = market_state(conn, symbol)
    console.print(state.render(), markup=False, highlight=False, soft_wrap=True)
    if failed:
        console.print("[red]the build failed or was skipped; the state above "
                      "may be out of date[/red]")
        raise typer.Exit(code=1)


@app.command()
def status():
    """Show row counts, last candle, and last quality verdict per symbol."""
    _setup_logging()
    settings = get_settings()
    now = datetime.now(timezone.utc)
    table = Table("symbol", "candles", "last candle (UTC)", "age", "last verdict")
    with connect() as conn:
        for s in settings.symbols:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM candles_1m WHERE symbol=%s", (s,))
                count = cur.fetchone()[0]
                cur.execute(
                    "SELECT verdict FROM data_quality_reports WHERE symbol=%s "
                    "ORDER BY created_at DESC LIMIT 1", (s,)
                )
                row = cur.fetchone()
            last = last_candle_time(conn, s)
            if last is None:
                age = "-"
            else:
                minutes = int((now - last).total_seconds() // 60)
                # More than STALE_AFTER_MINUTES behind means the live
                # collector is down or the feed is stuck.
                age = (f"[red]{minutes}m STALE[/red]"
                       if minutes > STALE_AFTER_MINUTES else f"{minutes}m")
            table.add_row(
                s, f"{count:,}",
                last.strftime("%Y-%m-%d %H:%M") if last else "-",
                age,
                row[0] if row else "-",
            )
    console.print(table)



def _ml_data(conn, symbols: list[str]) -> dict:
    """Quality gate and freshness check, then every symbol's training data.
    Prints the reason and exits 1 when training must not proceed."""
    try:
        assert_trainable(conn, symbols, "1m")
        check_fresh(conn, symbols)
    except DataQualityError as exc:
        console.print(escape(str(exc)), soft_wrap=True)
        raise typer.Exit(code=1)
    except StaleFeaturesError as exc:
        console.print(f"{escape(str(exc))}\nrun `tb features build` first",
                      soft_wrap=True)
        raise typer.Exit(code=1)
    return {s: load_symbol_data(conn, s) for s in symbols}


def _data_end(tau, folds, horizon: int) -> datetime:
    """The newest time whose price a run used: the last test label's end."""
    last = max(tau[f.test_idx].max() for f in folds)
    return (last + timedelta(hours=horizon)).to_pydatetime()


def _target(name: str):
    try:
        return get_target(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


TARGET_HELP = "What to predict: " + ", ".join(TARGETS) + " (default move3)"


def _horizons(values: list[int] | None) -> tuple[int, ...]:
    if not values:
        return HORIZONS
    bad = [h for h in values if h not in HORIZONS]
    if bad:
        raise typer.BadParameter(
            f"unknown horizon {bad[0]}; expected one of "
            + ", ".join(map(str, HORIZONS)))
    return tuple(values)


@ml_app.command("evaluate")
def ml_evaluate_cmd(
    horizon: list[int] = typer.Option(
        None, "--horizon", help="Hours ahead (1, 4 or 24); repeatable; default all"),
    holdout: bool = typer.Option(
        False, "--holdout", help="Test on the final holdout period (counted)"),
    target: str = typer.Option("move3", "--target", help=TARGET_HELP),
):
    """Walk-forward test of every model; stores the run and prints a report."""
    tgt = _target(target)
    horizons = _horizons(horizon)
    _setup_logging()
    symbols = get_settings().symbols
    began = time.monotonic()

    def say(text: str) -> None:
        minutes, seconds = divmod(int(time.monotonic() - began), 60)
        console.print(f"[{minutes:02d}:{seconds:02d}] {text}", markup=False,
                      highlight=False, soft_wrap=True)

    with connect() as conn:
        say("checking data quality and loading features ...")
        data = _ml_data(conn, symbols)
        say("data loaded")
        for h in horizons:
            ds = build_dataset(data, h, tuple(symbols), tgt)
            if holdout:
                fold = holdout_fold(ds.tau, h, ML_FOLDS)
                folds = [fold] if fold else []
            else:
                folds = walk_forward_folds(ds.tau, h, ML_FOLDS)
            if not folds:
                console.print(f"next {h}h: no test rows; nothing evaluated")
                continue
            say(f"next {h}h: {len(ds.X):,} rows, {len(folds)} test periods")
            result = ml_evaluate(ds, h, folds, progress=say, classes=tgt.classes)
            say(f"next {h}h: saving {len(result.predictions):,} predictions")
            run_id = ml_store.save_run(
                conn, kind="holdout" if holdout else "walk_forward", horizon=h,
                symbols=symbols, data_end=_data_end(ds.tau, folds, h),
                config=run_config(h, ML_FOLDS, tgt.name), metrics=result.metrics,
                predictions=result.predictions, target=tgt.name)
            conn.commit()
            console.print(render_report(ml_store.load_run(conn, run_id)),
                          markup=False, highlight=False, soft_wrap=True)
            console.print()
        say("done")


@ml_app.command("runs")
def ml_runs_cmd():
    """List stored evaluation runs."""
    with connect() as conn:
        runs = ml_store.list_runs(conn)
    table = Table("run", "target", "kind", "horizon", "created (UTC)",
                  "predictions", "xgb skill")
    for r in runs:
        skill = "-" if r["xgb_skill"] is None else f"{r['xgb_skill']:+.1%}"
        table.add_row(str(r["run_id"]), r["target"], r["kind"], f"{r['horizon']}h",
                      f"{r['created_at']:%Y-%m-%d %H:%M}",
                      f"{r['n_predictions']:,}", skill)
    console.print(table)


@ml_app.command("report")
def ml_report_cmd(run_id: int = typer.Argument(..., help="From tb ml runs")):
    """Print a stored run's report again."""
    with connect() as conn:
        run = ml_store.load_run(conn, run_id)
    if run is None:
        console.print(f"[red]no run {run_id}[/red]")
        raise typer.Exit(code=1)
    console.print(render_report(run), markup=False, highlight=False, soft_wrap=True)


@ml_app.command("train")
def ml_train_cmd(target: str = typer.Option("move3", "--target", help=TARGET_HELP)):
    """Fit the prediction models on all labelled history and save them."""
    tgt = _target(target)
    _setup_logging()
    symbols = get_settings().symbols
    trained_at = datetime.now(timezone.utc)
    with connect() as conn:
        data = _ml_data(conn, symbols)
        for h in HORIZONS:
            ds = build_dataset(data, h, tuple(symbols), tgt)
            fit, cal = final_split(ds.tau, h, ML_FOLDS.cal_fraction)
            run_id = ml_store.latest_run_id(conn, h, target=tgt.name)
            for model in make_models(tgt.n_classes):
                if model.name not in PREDICT_MODELS:
                    continue
                model.fit(ds.X.iloc[fit], ds.y[fit], ds.X.iloc[cal], ds.y[cal])
                path = save_model(model, h, {
                    "symbols": list(symbols), "run_id": run_id,
                    "train_start": ds.tau.min().isoformat(),
                    "train_end": ds.tau.max().isoformat(),
                    "trained_at": trained_at.isoformat(),
                }, target=tgt.name)
                console.print(f"{tgt.name} next {h}h {model.name}: {len(fit) + len(cal):,} rows "
                              f"to {ds.tau.max():%Y-%m-%d %H:%M} UTC -> {path}",
                              markup=False, soft_wrap=True)


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="For example BTCUSDT"),
    no_build: bool = typer.Option(
        False, "--no-build", help="Use the stored features without building"),
):
    """Print the current probabilities for one symbol (not advice)."""
    _setup_logging()
    symbol = symbol.upper()
    with connect() as conn:
        if last_candle_time(conn, symbol) is None:
            console.print(f"[red]{escape(symbol)}: no 1m candles stored; "
                          "run tb backfill first[/red]", soft_wrap=True)
            raise typer.Exit(code=1)
        failed = False
        if not no_build:
            failed = _build_symbols(conn, [symbol], FEATURE_TIMEFRAMES, False)
        try:
            state = prediction_state(conn, symbol)
        except NoRecentFeatures as exc:
            console.print(f"{exc}\nrun `tb features build` first", markup=False,
                          soft_wrap=True)
            raise typer.Exit(code=1)
        except FileNotFoundError:
            console.print("no saved models; run `tb ml train` first", markup=False)
            raise typer.Exit(code=1)
        except ArtifactMismatch as exc:
            console.print(f"{exc}\nrun `tb ml train` again", markup=False,
                          soft_wrap=True)
            raise typer.Exit(code=1)
    console.print(render_prediction(state), markup=False, highlight=False,
                  soft_wrap=True)
    if failed:
        console.print("[red]the build failed or was skipped; the probabilities "
                      "above may be out of date[/red]")
        raise typer.Exit(code=1)


@ml_app.command("diagnose")
def ml_diagnose_cmd(run_id: int = typer.Argument(..., help="From tb ml runs")):
    """Split a run's skill into size (move vs flat) and direction (up vs down)."""
    with connect() as conn:
        run = ml_store.load_run(conn, run_id)
        if run is None:
            console.print(f"[red]no run {run_id}[/red]")
            raise typer.Exit(code=1)
        if run["target"] != "move3":
            console.print(f"run {run_id} is a {run['target']} run; size vs direction "
                          "is only for move3 runs (vol3 and dir2 already ask those "
                          "questions separately)", markup=False, soft_wrap=True)
            raise typer.Exit(code=1)
        pred = ml_store.load_predictions(conn, run_id)
    console.print(render_diagnosis(run_id, run["horizon"], split_skill(pred)),
                  markup=False, highlight=False, soft_wrap=True)


@app.command()
def dashboard(port: int = typer.Option(8050, help="Local port")):
    """Open a read-only dashboard in your browser (on this PC only)."""
    import uvicorn

    from dashboard.app import create_app

    _setup_logging()
    console.print(f"Dashboard: http://127.0.0.1:{port}  (Ctrl+C to stop)",
                  markup=False)
    uvicorn.run(create_app(connect), host="127.0.0.1", port=port,
                log_level="warning")


@backtest_app.command("grid")
def backtest_grid(
    symbol: str = typer.Option(None, help="One symbol; default is all configured"),
    run: int = typer.Option(None, "--run", help="A walk-forward vol3 run with "
                            "horizon 24; default the newest"),
    fee: float = typer.Option(0.001, help="Fee per fill (0.001 = 0.1%)"),
):
    """Spot grid bots: always on, paused on forecast-wild days, or paused at
    random, against cash and hold."""
    from backtest.engine import SLIPPAGE, STRATEGIES, run_strategy
    from backtest.load import check_run, default_run, load_coin
    from backtest.report import render

    _setup_logging()
    symbols = [symbol.upper()] if symbol else get_settings().symbols
    with connect() as conn:
        run_id = run if run is not None else default_run(conn)
        if run_id is None:
            console.print("no vol3 run with horizon 24; run "
                          "`tb ml evaluate --target vol3` first", markup=False)
            raise typer.Exit(code=1)
        problem = check_run(conn, run_id)
        if problem:
            console.print(problem, markup=False, soft_wrap=True)
            raise typer.Exit(code=1)
        coins = {}
        for s in symbols:
            console.print(f"{s}: loading and simulating ...", markup=False)
            data = load_coin(conn, s, run_id)
            coins[s] = {name: run_strategy(data, name, fee, SLIPPAGE)
                        for name in STRATEGIES}
    console.print(render(run_id, fee, SLIPPAGE, coins), markup=False,
                  highlight=False, soft_wrap=True)


@app.command("signals")
def signals_cmd(
    no_build: bool = typer.Option(
        False, "--no-build", help="Use the stored features without building"),
):
    """Grid conditions per coin: WAIT or normal, with every reason (not advice)."""
    from signals import collect as signals_collect
    from signals.engine import error_signal, evaluate
    from signals.render import render as render_signals
    from signals.settings import load_settings
    from signals.store import save_signal

    _setup_logging()
    symbols = get_settings().symbols
    failed, problems, sigs = False, {}, []
    with connect() as conn:
        settings = load_settings(conn)
        for s in symbols:
            # one coin's error must not stop the others (as _build_symbols)
            try:
                if not no_build and last_candle_time(conn, s) is not None:
                    failed = _build_symbols(conn, [s], FEATURE_TIMEFRAMES, False,
                                            problems) or failed
                problem = problems.get(s)
                if no_build or (problem and problem.startswith("error: ")):
                    quality = None             # the quality gate did not finish
                else:
                    quality = problem == "data quality FAIL"
                build_problem = None if problem in (None, "data quality FAIL") else problem
                sig = evaluate(signals_collect.collect(conn, s, quality,
                                                       build_problem=build_problem),
                               settings)
            except Exception as exc:
                try:
                    conn.rollback()
                except Exception:
                    pass  # a broken connection must not mask the original error
                logging.getLogger(__name__).error("%s: signal failed: %s", s, exc)
                failed = True
                sig = error_signal(s, str(exc), settings)
            try:
                save_signal(conn, sig)
            except Exception as exc:
                try:
                    conn.rollback()
                except Exception:
                    pass
                console.print(f"{s}: could not store the signal: {exc}",
                              markup=False, soft_wrap=True)
                failed = True
            sigs.append(sig)
    console.print(render_signals(sigs, settings, datetime.now(timezone.utc)),
                  markup=False, highlight=False, soft_wrap=True)
    if failed:
        console.print("[red]a feature build or a coin's signal failed or was "
                      "skipped; see the lines above[/red]")
        raise typer.Exit(code=1)


@risk_app.command("show")
def risk_show():
    """Print your risk settings."""
    from signals.render import settings_lines
    from signals.settings import load_settings

    with connect() as conn:
        settings = load_settings(conn)
    console.print("\n".join(settings_lines(settings)), markup=False, highlight=False)


@risk_app.command("set")
def risk_set(
    profile: str = typer.Option(None, help="defensive or return"),
    wild_threshold: float = typer.Option(
        None, "--wild-threshold", help="WAIT when P(wild) is at or above this (0.05 to 0.95)"),
    capital: float = typer.Option(None, "--capital", help="Your grid capital in USDT"),
    no_capital: bool = typer.Option(False, "--no-capital", help="Forget the capital"),
    max_coin_share: float = typer.Option(
        None, "--max-coin-share", help="Largest share of capital per coin (0.01 to 1)"),
    fee: float = typer.Option(None, help="Fee per fill (0.00075 = 0.075%)"),
):
    """Change some risk settings; the others stay as they are."""
    from dataclasses import replace

    from signals.render import settings_lines
    from signals.settings import load_settings, save_settings

    changes = {k: v for k, v in dict(profile=profile, wild_threshold=wild_threshold,
                                     capital_usdt=capital, max_coin_share=max_coin_share,
                                     fee=fee).items() if v is not None}
    if no_capital:
        if capital is not None:
            console.print("give --capital or --no-capital, not both", markup=False)
            raise typer.Exit(code=1)
        changes["capital_usdt"] = None
    with connect() as conn:
        settings = replace(load_settings(conn), **changes)
        try:
            save_settings(conn, settings)
        except ValueError as exc:
            console.print(f"not saved: {exc}", markup=False, soft_wrap=True)
            raise typer.Exit(code=1)
    console.print("\n".join(settings_lines(settings)), markup=False, highlight=False)


PROFIT_HELP = "The bot's Total profit in USDT as the Binance app shows it (not Grid profit)"
TIME_HELP = "YYYY-MM-DD HH:MM, your PC's local time (or UTC with --utc)"


def _bots_fail(message: str):
    console.print(message, markup=False, soft_wrap=True)
    raise typer.Exit(code=1)


def _typed_time(text: str | None, utc: bool) -> tuple[datetime, str]:
    """The UTC time to store, and a note on what was typed."""
    from bots.times import parse_time

    if text is None:
        return datetime.now(timezone.utc).replace(second=0, microsecond=0), ""
    try:
        t = parse_time(text, utc)
    except ValueError as exc:
        _bots_fail(str(exc))
    return t, "" if utc else f" (you typed {text.strip()} local time)"


def _bot_or_fail(conn, bot_id: int) -> dict:
    from bots.store import get_bot

    bot = get_bot(conn, bot_id)
    if bot is None:
        _bots_fail(f"no bot {bot_id} (see tb bots list)")
    return bot


@bots_app.command("add")
def bots_add(
    symbol: str = typer.Option(..., help="The coin against USDT, for example BTCUSDT"),
    invest: float = typer.Option(..., help="USDT put into the bot"),
    start: str = typer.Option(..., help="When the bot started: " + TIME_HELP),
    utc: bool = typer.Option(False, "--utc", help="Times are UTC, not local time"),
    name: str = typer.Option(None, help="A name; default '<SYMBOL> grid'"),
    low: float = typer.Option(None, help="Lower price of the grid range"),
    high: float = typer.Option(None, help="Upper price of the grid range"),
    grids: int = typer.Option(None, help="Number of grids"),
):
    """Record a bot you started on Binance (funded with USDT)."""
    from bots.store import add_bot
    from signals.settings import load_settings

    t, typed = _typed_time(start, utc)
    with connect() as conn:
        fee = load_settings(conn).fee
        try:
            bot_id = add_bot(conn, symbol=symbol.upper(), invest=invest, start=t,
                             fee=fee, now=datetime.now(timezone.utc), name=name,
                             low=low, high=high, grids=grids)
        except ValueError as exc:
            _bots_fail(f"not stored: {exc}")
    console.print(f"bot {bot_id} recorded: {name or symbol.upper() + ' grid'}, "
                  f"{invest:,.2f} USDT, started {t:%Y-%m-%d %H:%M} UTC{typed}; "
                  f"holding is compared with a {fee:.3%} fee",
                  markup=False, soft_wrap=True)


def _bots_reading(bot_id: int, profit: float, at: str | None, utc: bool,
                  final: bool) -> None:
    from bots.report import reading_line
    from bots.store import add_update
    from bots.summary import history

    t, typed = _typed_time(at, utc)
    with connect() as conn:
        _bot_or_fail(conn, bot_id)
        try:
            add_update(conn, bot_id, t, profit, final=final,
                       now=datetime.now(timezone.utc))
        except ValueError as exc:
            _bots_fail(f"not stored: {exc}")
        last = history(conn, bot_id)[-1]
    what = "stop stored; the bot is now stopped" if final else "update stored"
    console.print(f"{what} for bot {bot_id} at {t:%Y-%m-%d %H:%M} UTC{typed}",
                  markup=False, soft_wrap=True)
    console.print(reading_line(last), markup=False, soft_wrap=True)


@bots_app.command("update")
def bots_update(
    bot_id: int = typer.Argument(..., help="From tb bots list"),
    profit: float = typer.Option(..., help=PROFIT_HELP),
    at: str = typer.Option(None, help="When you read it: " + TIME_HELP + "; default now"),
    utc: bool = typer.Option(False, "--utc", help="Times are UTC, not local time"),
):
    """Record a bot's current Total profit."""
    _bots_reading(bot_id, profit, at, utc, final=False)


@bots_app.command("stop")
def bots_stop(
    bot_id: int = typer.Argument(..., help="From tb bots list"),
    profit: float = typer.Option(..., help="The final " + PROFIT_HELP),
    at: str = typer.Option(None, help="When it stopped: " + TIME_HELP + "; default now"),
    utc: bool = typer.Option(False, "--utc", help="Times are UTC, not local time"),
):
    """Record that a bot stopped, with its final Total profit."""
    _bots_reading(bot_id, profit, at, utc, final=True)


@bots_app.command("undo")
def bots_undo(bot_id: int = typer.Argument(..., help="From tb bots list")):
    """Remove a bot's latest update (a stop update reopens the bot)."""
    from bots.store import undo

    with connect() as conn:
        _bot_or_fail(conn, bot_id)
        removed = undo(conn, bot_id)
    if removed is None:
        console.print(f"bot {bot_id} has no updates to remove", markup=False)
        return
    again = "; the bot is running again" if removed["final"] else ""
    console.print(f"removed the update of {removed['at']:%Y-%m-%d %H:%M} UTC "
                  f"(Total profit {removed['total_profit']:,.2f} USDT){again}",
                  markup=False, soft_wrap=True)


@bots_app.command("delete")
def bots_delete(
    bot_id: int = typer.Argument(..., help="From tb bots list"),
    yes: bool = typer.Option(False, "--yes", help="Confirm: removes the bot and all its updates"),
):
    """Remove a bot and all its updates."""
    from bots.store import delete_bot

    with connect() as conn:
        bot = _bot_or_fail(conn, bot_id)
        if not yes:
            _bots_fail(f"this removes bot {bot_id} ({bot['name']}) and all its "
                       "updates; run it again with --yes to confirm")
        delete_bot(conn, bot_id)
    console.print(f"bot {bot_id} ({bot['name']}) removed", markup=False)


@bots_app.command("list")
def bots_list():
    """Every bot at its latest update against holding, and all bots together."""
    from bots.compare import totals
    from bots.report import render_list
    from bots.summary import bot_rows

    with connect() as conn:
        rows = bot_rows(conn)
    t = totals([(r["invest_usdt"], r["comparison"]) for r in rows])
    console.print(render_list(rows, t), markup=False, highlight=False, soft_wrap=True)


@bots_app.command("show")
def bots_show(bot_id: int = typer.Argument(..., help="From tb bots list")):
    """One bot's full update history against holding."""
    from bots.report import render_show
    from bots.summary import bot_row, history

    with connect() as conn:
        bot = bot_row(conn, _bot_or_fail(conn, bot_id))
        hist = history(conn, bot_id)
    console.print(render_show(bot, hist), markup=False, highlight=False, soft_wrap=True)


if __name__ == "__main__":
    app()
