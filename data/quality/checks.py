import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from data.quality.report import Report
from data.storage.repository import _floor, find_gaps

log = logging.getLogger(__name__)

STEP = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15), "1h": timedelta(hours=1),
        "4h": timedelta(hours=4), "1d": timedelta(days=1)}

# How long after a bucket closes the refresh policy may still be yet to
# materialize it: the policy's schedule_interval + its end_offset (1 minute)
# + 5 minutes of grace. A test pins these to timescaledb_information.jobs.
POLICY_LAG = {"5m": timedelta(minutes=11), "15m": timedelta(minutes=21),
              "1h": timedelta(minutes=66), "4h": timedelta(minutes=66),
              "1d": timedelta(minutes=66)}

PASS_COMPLETENESS = 99.9
WARN_COMPLETENESS = 95.0


class DataQualityError(Exception):
    pass


# Every candle must satisfy this arithmetic; a row matching the predicate is
# invalid. Evaluated in SQL so the history never has to come into Python.
INVALID_PREDICATE = """
    open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
    OR volume IS NULL OR volume < 0 OR trade_count < 0
    OR high < low OR high < open OR high < close
    OR low > open OR low > close
"""


# Any single gap longer than this that nothing explains is an ingestion hole
# too big to train through: FAIL. Shorter unexplained gaps are WARN.
MAX_UNEXPLAINED_GAP = timedelta(minutes=60)


def _month_starts(start: datetime, end: datetime) -> list[datetime]:
    """First instants of every calendar month overlapping [start, end)."""
    last = end - timedelta(microseconds=1)
    y, m = start.year, start.month
    out = []
    while (y, m) <= (last.year, last.month):
        out.append(datetime(y, m, 1, tzinfo=timezone.utc))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def classify_gaps(gaps, loaded_months, known_outages=None):
    """Split gaps into (explained, unexplained).

    A gap is explained -- the exchange's own outage, not ours -- when every
    month it touches had its monthly archive loaded with rows > 0 (that
    archive is Binance's own record of the month), or when a known outage
    covers it. Any other gap is an ingestion hole.
    """
    explained, unexplained = [], []
    for gap in gaps:
        g_start, g_end = gap
        by_outage = any(o_start <= g_start and g_end <= o_end
                        for o_start, o_end in known_outages or ())
        by_archive = all(m in loaded_months for m in _month_starts(g_start, g_end))
        (explained if by_outage or by_archive else unexplained).append(gap)
    return explained, unexplained


def verdict_for(completeness: float, invalid: int, duplicates: int,
                unexplained_gaps: int = 0,
                longest_unexplained: timedelta = timedelta(0)) -> str:
    if invalid or duplicates:
        return "FAIL"
    if longest_unexplained > MAX_UNEXPLAINED_GAP:
        return "FAIL"
    if completeness < WARN_COMPLETENESS:
        return "FAIL"
    if unexplained_gaps or completeness < PASS_COMPLETENESS:
        return "WARN"
    return "PASS"


def _loaded_months(conn, symbol: str) -> set[datetime]:
    """Months whose monthly archive loaded with rows > 0."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT period_start FROM ingestion_runs "
            "WHERE component = 'backfill' AND symbol = %s "
            "AND status = 'success' AND rows_written > 0", (symbol,)
        )
        return {r[0] for r in cur.fetchall()}


def _has_1m_rows(conn, symbol: str, gap) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM candles_1m WHERE symbol = %s "
                    "AND open_time >= %s AND open_time < %s)", (symbol, *gap))
        return cur.fetchone()[0]


def _gap_detail(gap, explained: bool) -> dict:
    g_start, g_end = gap
    return {"start": g_start.isoformat(), "end": g_end.isoformat(),
            "minutes": int((g_end - g_start).total_seconds() // 60),
            "explained": explained}


def run_quality_checks(conn, symbol: str, timeframe: str = "1m",
                       start: datetime | None = None,
                       end: datetime | None = None,
                       known_outages: list[tuple] | None = None,
                       now: datetime | None = None) -> Report:
    step = STEP[timeframe]
    table = "candles_1m" if timeframe == "1m" else f"candles_{timeframe}"
    with conn.cursor() as cur:
        # The expected range always comes from the 1m truth, never from the
        # aggregate itself: a view that holds only its policy's recent window
        # must not get to define its own range and pass.
        if timeframe == "1m":
            cur.execute("SELECT min(open_time), max(open_time), count(*) "
                        "FROM candles_1m WHERE symbol = %s", (symbol,))
        else:
            # First bucket touched by 1m data, through the last complete one.
            cur.execute(
                "SELECT time_bucket(%(step)s, min(open_time)), "
                "time_bucket(%(step)s, max(open_time) + interval '1 minute') "
                "- %(step)s, count(*) FROM candles_1m WHERE symbol = %(symbol)s",
                {"step": step, "symbol": symbol},
            )
        first, last, total = cur.fetchone()
        default_end = last
        if timeframe != "1m" and total:
            # The refresh policy materializes a closed bucket only some
            # minutes later; until then it is not yet a hole. The newest
            # bucket that must exist is the last one that closed at least
            # POLICY_LAG ago.
            now = now or datetime.now(timezone.utc)
            default_end = min(last, _floor(now - POLICY_LAG[timeframe], step) - step)
        if total and (last < first or (end is None and default_end < first)):
            total = 0  # 1m data exists, but not one complete, due bucket of it
        if not total:
            report = Report(symbol=symbol, timeframe=timeframe,
                            checked_from=None, checked_to=None,
                            total_candles=0, duplicates=0, invalid=0,
                            missing=0, completeness_pct=0.0, verdict="FAIL",
                            details={"reason": "no data"})
            _store(conn, report)
            return report

        start = start or first
        end = end or default_end
        # Counts, invalid rows and non-unique timestamps in one SQL pass;
        # only three integers come back to Python, never the candles.
        cur.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {INVALID_PREDICATE}), "
            f"count(*) - count(DISTINCT open_time) "
            f"FROM {table} WHERE symbol = %s AND open_time BETWEEN %s AND %s",
            (symbol, start, end),
        )
        in_range, invalid, out_of_order = cur.fetchone()

        # The primary key makes duplicates impossible in candles_1m; this
        # check exists so a future schema change cannot silently allow them.
        cur.execute(
            f"SELECT count(*) FROM (SELECT symbol, open_time FROM {table} "
            f"WHERE symbol = %s GROUP BY 1, 2 HAVING count(*) > 1) d", (symbol,)
        )
        duplicates = cur.fetchone()[0]

    # Gaps via lag() in SQL (see find_gaps); only the gaps come back.
    gaps = find_gaps(conn, symbol, start, end + step, step=step, table=table)
    # A missing bucket where 1m candles exist is an aggregate that disagrees
    # with the 1m truth -- never an exchange outage, whatever the month.
    aggregate_holes = ([g for g in gaps if _has_1m_rows(conn, symbol, g)]
                       if timeframe != "1m" else [])
    explained, unexplained = classify_gaps(
        [g for g in gaps if g not in aggregate_holes],
        _loaded_months(conn, symbol), known_outages,
    )
    unexplained = sorted(unexplained + aggregate_holes)
    missing = sum(int((g_end - g_start) / step) for g_start, g_end in unexplained)
    explained_missing = sum(int((g_end - g_start) / step)
                            for g_start, g_end in explained)
    longest = max((g_end - g_start for g_start, g_end in unexplained),
                  default=timedelta(0))

    expected = int((end - start) / step) + 1
    completeness = 100.0 * (expected - missing) / expected if expected else 0.0
    verdict = verdict_for(completeness, invalid, duplicates + out_of_order,
                          unexplained_gaps=len(unexplained),
                          longest_unexplained=longest)

    # Unexplained gaps first (they are what needs action), longest first.
    ranked = sorted(
        [(g, False) for g in unexplained] + [(g, True) for g in explained],
        key=lambda item: (item[1], -(item[0][1] - item[0][0]).total_seconds()),
    )
    report = Report(
        symbol=symbol, timeframe=timeframe, checked_from=start, checked_to=end,
        total_candles=in_range, duplicates=duplicates, invalid=invalid,
        missing=missing, completeness_pct=round(completeness, 6),
        verdict=verdict,
        details={
            "out_of_order": out_of_order,
            "gap_count": len(gaps),
            "unexplained_gap_count": len(unexplained),
            "aggregate_holes": len(aggregate_holes),
            "explained_missing": explained_missing,
            "longest_unexplained_minutes": int(longest.total_seconds() // 60),
            "largest_gaps": [_gap_detail(g, ex) for g, ex in ranked[:10]],
        },
    )
    _store(conn, report)
    return report


def _store(conn, report: Report) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO data_quality_reports (symbol, timeframe, checked_from, "
            "checked_to, total_candles, duplicates, invalid, missing, "
            "completeness_pct, verdict, details) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (report.symbol, report.timeframe, report.checked_from,
             report.checked_to, report.total_candles, report.duplicates,
             report.invalid, report.missing, report.completeness_pct,
             report.verdict, json.dumps(report.details)),
        )
    conn.commit()


def assert_trainable(conn, symbols: list[str], timeframe: str = "1m") -> list[Report]:
    """Every later sub-project calls this before training. A FAIL stops it.

    Returns the WARN reports, each already logged at WARNING level: a WARN
    does not block training, but it must never pass silently.
    """
    failures, warnings = [], []
    for symbol in symbols:
        report = run_quality_checks(conn, symbol, timeframe)
        if report.verdict == "FAIL":
            failures.append(report)
        elif report.verdict == "WARN":
            warnings.append(report)
            log.warning("data quality WARN for %s [%s]; training proceeds, "
                        "but look at this:%s", report.symbol, report.timeframe,
                        report.render())
    if failures:
        summary = "\n".join(r.render() for r in failures)
        raise DataQualityError(
            f"data quality FAILED for {len(failures)} symbol(s); "
            f"refusing to train:\n{summary}"
        )
    return warnings
