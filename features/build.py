"""Building feature rows for one symbol (spec sections 5.2 and 5.3).

`build_features` does one (symbol, timeframe): work out the cutoff, refresh
the continuous aggregates up to it, load the window, compute, write, commit.
`build_symbol` puts the 1m quality gate in front of all timeframes. The
build itself has no gate: the caller decides.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from data.quality.checks import STEP, run_quality_checks
from data.storage.repository import refresh_aggregates
from features.frame import build_cutoff, load_bars
from features.pipeline import FEATURE_TIMEFRAMES, compute_features, lookback_start
from features.store import has_stale_feature_set, last_built, upsert_features

log = logging.getLogger(__name__)

# Rows per upsert_features call. One call on a 1M-row full 5m build costs
# about 135 us per row and an estimated 2-3 GB of peak memory; batches keep
# the peak flat. All batches share one transaction, committed once at the end.
WRITE_BATCH_ROWS = 100_000


@dataclass
class BuildResult:
    symbol: str
    timeframe: str
    rows_written: int
    start: datetime | None       # first bar written
    end: datetime | None         # last bar written
    full: bool                   # full rebuild?
    seconds: float
    skipped_reason: str | None = None


def _first_1m(conn, symbol: str) -> datetime:
    with conn.cursor() as cur:
        cur.execute("SELECT min(open_time) FROM candles_1m WHERE symbol = %s",
                    (symbol,))
        return cur.fetchone()[0]


def _write_in_batches(conn, symbol: str, timeframe: str, frame) -> int:
    written = 0
    for i in range(0, len(frame), WRITE_BATCH_ROWS):
        written += upsert_features(conn, symbol, timeframe,
                                   frame.iloc[i:i + WRITE_BATCH_ROWS])
    return written


def build_features(conn, symbol: str, timeframe: str, *,
                   rebuild: bool = False) -> BuildResult:
    """Build the feature rows that are missing for (symbol, timeframe).

    Full when `rebuild`, when nothing is stored, or when any stored row has a
    different FEATURE_SET; otherwise incremental (spec section 5.3). Commits.
    """
    began = time.monotonic()
    step = STEP[timeframe]

    def result(rows=0, first=None, last=None, full=False, reason=None):
        return BuildResult(symbol, timeframe, rows, first, last, full,
                           time.monotonic() - began, reason)

    try:
        cutoff = build_cutoff(conn, symbol, timeframe)
        if cutoff is None:
            conn.commit()
            return result(reason="no 1m candles")
        built = last_built(conn, symbol, timeframe)
        full = rebuild or built is None or has_stale_feature_set(
            conn, symbol, timeframe)
        load_from = None if full else lookback_start(built, step)
        refresh_from = _first_1m(conn, symbol) if full else load_from

        # refresh_aggregates opens its own connection and needs this
        # connection's read transaction ended first.
        conn.commit()
        refresh_aggregates(conn, refresh_from, cutoff)

        bars = load_bars(conn, symbol, timeframe, load_from, cutoff)
        features = compute_features(bars, step)
        if not full:
            features = features[features.index > built]

        written = _write_in_batches(conn, symbol, timeframe, features)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    if written == 0:
        return result(full=full)
    return result(written, features.index[0].to_pydatetime(),
                  features.index[-1].to_pydatetime(), full)


def build_symbol(conn, symbol: str,
                 timeframes: tuple[str, ...] = FEATURE_TIMEFRAMES, *,
                 rebuild: bool = False) -> list[BuildResult]:
    """Run the 1m quality gate once, then build each timeframe.

    FAIL skips every timeframe; WARN logs and builds.
    """
    report = run_quality_checks(conn, symbol)
    if report.verdict == "FAIL":
        log.error("%s: 1m data quality FAIL, no features built", symbol)
        return [BuildResult(symbol, tf, 0, None, None, False, 0.0,
                            "data quality FAIL") for tf in timeframes]
    if report.verdict == "WARN":
        log.warning("%s: 1m data quality WARN (%.3f%% complete); building "
                    "anyway", symbol, report.completeness_pct)
    return [build_features(conn, symbol, tf, rebuild=rebuild)
            for tf in timeframes]
