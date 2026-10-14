"""Each bot with its comparison against holding, read from the database."""
from datetime import timedelta

from bots import store
from bots.compare import Comparison, compare
from bots.prices import price_at


def _missing(symbol: str, t) -> str:
    return (f"no stored price for {symbol} near {t:%Y-%m-%d %H:%M} UTC "
            "(run `tb backfill`, or add the coin to SYMBOLS)")


def _compare(conn, bot: dict, update: dict, p_start: float | None
             ) -> tuple[Comparison | None, str | None]:
    if p_start is None:
        return None, _missing(bot["symbol"], bot["start_time"])
    p_end = price_at(conn, bot["symbol"], update["at"])
    if p_end is None:
        return None, _missing(bot["symbol"], update["at"])
    return compare(bot["invest_usdt"], bot["fee"], update["total_profit"],
                   p_start, p_end), None


def _days(bot: dict, at) -> float:
    return (at - bot["start_time"]) / timedelta(days=1)


def history(conn, bot_id: int) -> list[dict]:
    bot = store.get_bot(conn, bot_id)
    if bot is None:
        return []
    p_start = price_at(conn, bot["symbol"], bot["start_time"])
    out = []
    for u in store.updates(conn, bot_id):
        comparison, note = _compare(conn, bot, u, p_start)
        out.append(dict(at=u["at"], total_profit=u["total_profit"], final=u["final"],
                        days=_days(bot, u["at"]), comparison=comparison, note=note))
    return out


def bot_row(conn, bot: dict) -> dict:
    """The bot at its latest update."""
    ups = store.updates(conn, bot["id"])
    row = {**bot, "status": "stopped" if bot["stopped_at"] else "running",
           "at": None, "total_profit": None, "days": None, "comparison": None,
           "note": None}
    if not ups:
        row["note"] = f"no update yet (tb bots update {bot['id']} --profit X)"
        return row
    last = ups[-1]
    p_start = price_at(conn, bot["symbol"], bot["start_time"])
    comparison, note = _compare(conn, bot, last, p_start)
    row.update(at=last["at"], total_profit=last["total_profit"],
               days=_days(bot, last["at"]), comparison=comparison, note=note)
    return row


def bot_rows(conn) -> list[dict]:
    return [bot_row(conn, b) for b in store.list_bots(conn)]
