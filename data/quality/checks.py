import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from data.quality.report import Report
from data.storage.repository import find_gaps

STEP = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15), "1h": timedelta(hours=1),
        "4h": timedelta(hours=4), "1d": timedelta(days=1)}

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


def subtract_known_outages(gaps, outages) -> list[tuple[datetime, datetime]]:
    """Drop gaps fully covered by a known exchange outage."""
    remaining = []
    for start, end in gaps:
        if any(o_start <= start and end <= o_end for o_start, o_end in outages):
            continue
        remaining.append((start, end))
    return remaining


def verdict_for(completeness: float, invalid: int, duplicates: int) -> str:
    if invalid or duplicates:
        return "FAIL"
    if completeness >= PASS_COMPLETENESS:
        return "PASS"
    if completeness >= WARN_COMPLETENESS:
        return "WARN"
    return "FAIL"


def run_quality_checks(conn, symbol: str, timeframe: str = "1m",
                       start: datetime | None = None,
                       end: datetime | None = None,
                       known_outages: list[tuple] | None = None) -> Report:
    step = STEP[timeframe]
    table = "candles_1m" if timeframe == "1m" else f"candles_{timeframe}"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT min(open_time), max(open_time), count(*) FROM {table} "
            f"WHERE symbol = %s", (symbol,)
        )
        first, last, total = cur.fetchone()
        if not total:
            report = Report(symbol=symbol, timeframe=timeframe,
                            checked_from=None, checked_to=None,
                            total_candles=0, duplicates=0, invalid=0,
                            missing=0, completeness_pct=0.0, verdict="FAIL",
                            details={"reason": "no data"})
            _store(conn, report)
            return report

        start = start or first
        end = end or last
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
    if known_outages:
        gaps = subtract_known_outages(gaps, known_outages)
    missing = sum(int((g_end - g_start) / step) for g_start, g_end in gaps)

    expected = int((end - start) / step) + 1
    completeness = 100.0 * (expected - missing) / expected if expected else 0.0
    verdict = verdict_for(completeness, invalid, duplicates + out_of_order)

    report = Report(
        symbol=symbol, timeframe=timeframe, checked_from=start, checked_to=end,
        total_candles=in_range, duplicates=duplicates, invalid=invalid,
        missing=missing, completeness_pct=round(completeness, 6),
        verdict=verdict,
        details={
            "out_of_order": out_of_order,
            "gap_count": len(gaps),
            "largest_gaps": [
                [g[0].isoformat(), g[1].isoformat()]
                for g in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:10]
            ],
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


def assert_trainable(conn, symbols: list[str], timeframe: str = "1m") -> None:
    """Every later sub-project calls this before training. A FAIL stops it."""
    failures = []
    for symbol in symbols:
        report = run_quality_checks(conn, symbol, timeframe)
        if report.verdict == "FAIL":
            failures.append(report)
    if failures:
        summary = "\n".join(r.render() for r in failures)
        raise DataQualityError(
            f"data quality FAILED for {len(failures)} symbol(s); "
            f"refusing to train:\n{summary}"
        )
