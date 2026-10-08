"""Every database read the dashboard makes (spec section 3). Read-only.

Callers validate symbol and timeframe first: the timeframe picks a table
name, so it must be one of FEATURE_TIMEFRAMES.
"""
from datetime import datetime, timedelta

from data.quality.checks import STEP
from features.frame import build_cutoff
from features.pipeline import FEATURE_TIMEFRAMES

# tb status calls a 1m feed older than this stale.
STALE_AFTER = timedelta(minutes=5)


def _float(v):
    return None if v is None else float(v)


def candles(conn, symbol: str, timeframe: str, bars: int) -> dict:
    """The newest `bars` closed bars, oldest first, with EMA 200 and RSI 14,
    and the newest feature row's regimes."""
    if timeframe not in FEATURE_TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    cutoff = build_cutoff(conn, symbol, timeframe)
    if cutoff is None:
        return {"bars": [], "latest": None}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT c.open_time, c.open::float8, c.high::float8, c.low::float8, "
            f"c.close::float8, c.volume::float8, f.ema_200, f.rsi_14 "
            f"FROM candles_{timeframe} c LEFT JOIN features_{timeframe} f "
            f"ON f.symbol = c.symbol AND f.open_time = c.open_time "
            f"WHERE c.symbol = %s AND c.open_time < %s "
            f"ORDER BY c.open_time DESC LIMIT %s",
            (symbol, cutoff, bars))
        rows = cur.fetchall()[::-1]
        cur.execute(
            f"SELECT open_time, trend_regime, volatility_regime, adx_14, vol_pct_365d "
            f"FROM features_{timeframe} WHERE symbol = %s AND open_time < %s "
            f"ORDER BY open_time DESC LIMIT 1", (symbol, cutoff))
        newest = cur.fetchone()
    keys = ("open", "high", "low", "close", "volume", "ema_200", "rsi_14")
    out = [{"time": int(r[0].timestamp()),
            **{k: _float(v) for k, v in zip(keys, r[1:])}} for r in rows]
    latest = None
    if newest is not None:
        latest = {"bar_close": (newest[0] + STEP[timeframe]).isoformat(),
                  "trend_regime": newest[1], "volatility_regime": newest[2],
                  "adx_14": _float(newest[3]), "vol_pct_365d": _float(newest[4])}
    return {"bars": out, "latest": latest}


def health(conn, symbols: list[str], now: datetime) -> list[dict]:
    """Per symbol: newest 1m candle and its age, newest feature bar per
    timeframe, latest quality verdict."""
    out = []
    with conn.cursor() as cur:
        for symbol in symbols:
            cur.execute("SELECT max(open_time) FROM candles_1m WHERE symbol = %s",
                        (symbol,))
            last = cur.fetchone()[0]
            features = {}
            for tf in FEATURE_TIMEFRAMES:
                cur.execute(f"SELECT max(open_time) FROM features_{tf} "
                            "WHERE symbol = %s", (symbol,))
                t = cur.fetchone()[0]
                features[tf] = t.isoformat() if t else None
            cur.execute("SELECT verdict, created_at FROM data_quality_reports "
                        "WHERE symbol = %s ORDER BY created_at DESC, id DESC LIMIT 1",
                        (symbol,))
            q = cur.fetchone()
            age = None if last is None else now - (last + timedelta(minutes=1))
            out.append({
                "symbol": symbol,
                "last_1m": last.isoformat() if last else None,
                "age_minutes": None if age is None else int(age.total_seconds() // 60),
                "stale": None if age is None else age > STALE_AFTER,
                "features": features,
                "verdict": q[0] if q else None,
                "verdict_at": q[1].isoformat() if q and q[1] else None,
            })
    return out
