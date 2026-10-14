"""The user's bots and their profit readings (spec sections 2 and 5).

Every rule is checked here before anything is written; a broken rule
raises ValueError and stores nothing.
"""
import math
import re
from datetime import datetime

_SYMBOL = re.compile(r"^[A-Z0-9]+USDT$")
_BOT = ("id", "name", "symbol", "invest_usdt", "start_time", "fee", "range_low",
        "range_high", "grids", "stopped_at", "created_at")
_UPDATE = ("id", "bot_id", "at", "total_profit", "final", "created_at")


def _finite(name: str, v: float) -> None:
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a number, got {v!r}")


def add_bot(conn, *, symbol: str, invest: float, start: datetime, fee: float,
            now: datetime, name: str | None = None, low: float | None = None,
            high: float | None = None, grids: int | None = None) -> int:
    if not _SYMBOL.match(symbol):
        raise ValueError(f"symbol must be a coin against USDT in capitals, "
                         f"like BTCUSDT; got {symbol!r}")
    _finite("invest", invest)
    if invest <= 0:
        raise ValueError("invest must be more than 0 USDT")
    if start > now:
        raise ValueError(f"start {start:%Y-%m-%d %H:%M} UTC is in the future")
    if (low is None) != (high is None):
        raise ValueError("give both --low and --high, or neither")
    if low is not None:
        _finite("low", low), _finite("high", high)
        if not 0 < low < high:
            raise ValueError("the range needs 0 < low < high")
    if grids is not None and not 2 <= grids <= 500:
        raise ValueError("grids must be between 2 and 500")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bots (name, symbol, invest_usdt, start_time, fee, range_low, "
            "range_high, grids) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (name or f"{symbol} grid", symbol, invest, start, fee, low, high, grids))
        bot_id = cur.fetchone()[0]
    conn.commit()
    return bot_id


def get_bot(conn, bot_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(_BOT)} FROM bots WHERE id = %s", (bot_id,))
        row = cur.fetchone()
    return dict(zip(_BOT, row)) if row else None


def list_bots(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(_BOT)} FROM bots ORDER BY id")
        return [dict(zip(_BOT, r)) for r in cur.fetchall()]


def updates(conn, bot_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(_UPDATE)} FROM bot_updates WHERE bot_id = %s "
                    "ORDER BY at, id", (bot_id,))
        return [dict(zip(_UPDATE, r)) for r in cur.fetchall()]


def add_update(conn, bot_id: int, at: datetime, profit: float, *, final: bool,
               now: datetime) -> int:
    bot = get_bot(conn, bot_id)
    if bot is None:
        raise ValueError(f"no bot {bot_id} (see tb bots list)")
    if bot["stopped_at"] is not None:
        raise ValueError(f"bot {bot_id} is stopped; `tb bots undo {bot_id}` "
                         "removes its stop update")
    _finite("profit", profit)
    if profit <= -bot["invest_usdt"]:
        raise ValueError(f"a loss of {-profit:,.2f} USDT is more than the "
                         f"{bot['invest_usdt']:,.2f} USDT invested")
    if at > now:
        raise ValueError(f"{at:%Y-%m-%d %H:%M} UTC is in the future")
    if at < bot["start_time"]:
        raise ValueError(f"{at:%Y-%m-%d %H:%M} UTC is before the bot's start "
                         f"({bot['start_time']:%Y-%m-%d %H:%M} UTC)")
    prev = updates(conn, bot_id)
    if prev and at < prev[-1]["at"]:
        raise ValueError(f"{at:%Y-%m-%d %H:%M} UTC is before its previous update "
                         f"({prev[-1]['at']:%Y-%m-%d %H:%M} UTC)")
    with conn.cursor() as cur:
        cur.execute("INSERT INTO bot_updates (bot_id, at, total_profit, final) "
                    "VALUES (%s, %s, %s, %s) RETURNING id", (bot_id, at, profit, final))
        update_id = cur.fetchone()[0]
        if final:
            cur.execute("UPDATE bots SET stopped_at = %s WHERE id = %s", (at, bot_id))
    conn.commit()
    return update_id


def undo(conn, bot_id: int) -> dict | None:
    """Remove the bot's latest update (reopening the bot if it was the stop
    update). Returns the removed update, or None if there was none."""
    prev = updates(conn, bot_id)
    if not prev:
        return None
    last = prev[-1]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM bot_updates WHERE id = %s", (last["id"],))
        if last["final"]:
            cur.execute("UPDATE bots SET stopped_at = NULL WHERE id = %s", (bot_id,))
    conn.commit()
    return last


def delete_bot(conn, bot_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM bots WHERE id = %s", (bot_id,))
        gone = cur.rowcount > 0
    conn.commit()
    return gone
