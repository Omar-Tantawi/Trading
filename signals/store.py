"""Every signal shown is stored with the settings it used (spec section 4)."""
from psycopg.types.json import Jsonb

from signals.engine import Signal

_COLUMNS = ("id", "created_at", "symbol", "bar_open_time", "status", "p_wild",
            "reasons", "range_low", "range_high", "step_pct", "capital_limit",
            "settings")


def save_signal(conn, sig: Signal) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO signals (symbol, bar_open_time, status, p_wild, reasons, "
            "range_low, range_high, step_pct, capital_limit, settings) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (sig.symbol, sig.bar_open_time, sig.status, sig.p_wild, sig.reasons,
             sig.range_low, sig.range_high, sig.step_pct, sig.capital_limit,
             Jsonb(sig.settings)))
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def latest_signals(conn) -> list[dict]:
    """The newest signal of each coin, by symbol."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT ON (symbol) {', '.join(_COLUMNS)} FROM signals "
                    "ORDER BY symbol, created_at DESC, id DESC")
        return [dict(zip(_COLUMNS, row)) for row in cur.fetchall()]
