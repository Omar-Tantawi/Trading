import asyncio
import logging
import sys
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from data.collectors.backfill import ArchiveDownloader, backfill_symbol
from data.collectors.binance_rest import BinanceRest, parse_symbol_info
from data.collectors.live import LiveCollector
from data.config import get_settings
from data.quality.checks import STEP, run_quality_checks
from data.storage.db import connect, run_migrations
from data.storage.repository import last_candle_time, upsert_symbol

app = typer.Typer(help="AI Trading Buddy data foundation")
db_app = typer.Typer(help="Database maintenance")
symbols_app = typer.Typer(help="Symbol metadata")
app.add_typer(db_app, name="db")
app.add_typer(symbols_app, name="symbols")
console = Console()

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
        for s in targets:
            try:
                written = backfill_symbol(conn, s, downloader, api, start=start)
                console.print(f"{s}: {written:,} candles written")
            except Exception as exc:
                # One symbol's failure must not stop the others; the failure
                # is still surfaced via logging, console output, and a
                # non-zero exit code once every symbol has been attempted.
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
    with connect() as conn:
        for s in targets:
            console.print(run_quality_checks(conn, s, timeframe).render())


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


if __name__ == "__main__":
    app()
