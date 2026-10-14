"""Text for `tb bots list` and `tb bots show`. Describes; never advises."""
from bots.compare import Comparison, Totals

FOOTER = ("Bot results are the Total profit typed in from the Binance app; "
          "holding = the same USDT turned into the coin at the bot's start "
          "(one fee) and kept. Past results. Not advice.")


def _money(v: float) -> str:
    return f"{v:+,.2f} USDT"


def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _result(c: Comparison) -> str:
    return (f"bot {_money(c.bot_profit)} ({_pct(c.bot_pct)}), holding "
            f"{_money(c.hold_profit)} ({_pct(c.hold_pct)})")


def _verdict(verdict: str, diff: float, pts: float) -> str:
    if verdict == "level":
        return "bot level with holding"
    word = "ahead of" if verdict == "ahead" else "behind"
    return (f"bot {word} holding by {abs(diff):,.2f} USDT "
            f"({pts * 100:+.2f} pts)")


def _head(bot: dict) -> str:
    start = f"{bot['start_time']:%Y-%m-%d %H:%M} UTC"
    if bot["status"] == "running":
        since = f"running since {start}"
    else:
        stopped = bot.get("stopped_at")
        when = f" {stopped:%Y-%m-%d %H:%M} UTC" if stopped else ""
        since = f"stopped{when}, started {start}"
    return (f"#{bot['id']} {bot['name']} ({bot['symbol']}), {since}, "
            f"{bot['invest_usdt']:,.2f} USDT")


def _when(at, days: float) -> str:
    return f"at {at:%Y-%m-%d %H:%M} UTC ({days:.1f} days)"


def render_list(rows: list[dict], t: Totals) -> str:
    if not rows:
        return ("no bots yet; record one with tb bots add --symbol BTCUSDT "
                "--invest 500 --start \"YYYY-MM-DD HH:MM\"")
    lines = ["Bots vs holding the coin", ""]
    for r in rows:
        lines.append(_head(r))
        if r["comparison"] is not None:
            c = r["comparison"]
            lines.append(f"   {_when(r['at'], r['days'])}: {_result(c)}")
            lines.append("   " + _verdict(c.verdict, c.diff, c.diff_pts))
        else:
            if r["at"] is not None:
                lines.append(f"   {_when(r['at'], r['days'])}: bot "
                             f"{_money(r['total_profit'])}")
            lines.append(f"   {r['note']}")
    lines.append("")
    if t.n_compared:
        lines.append(f"All bots ({t.n_compared} of {t.n_bots} compared, "
                     f"{t.invested:,.2f} USDT invested): bot {_money(t.bot_profit)} "
                     f"({_pct(t.bot_pct)}), holding {_money(t.hold_profit)} "
                     f"({_pct(t.hold_pct)})")
        lines.append("   " + _verdict(t.verdict, t.diff, t.diff_pts))
    else:
        lines.append(f"All bots: none of the {t.n_bots} can be compared yet")
    lines += ["", FOOTER]
    return "\n".join(lines)


def reading_line(h: dict) -> str:
    """One update of a bot's history, against holding."""
    stop = " (stop)" if h["final"] else ""
    if h["comparison"] is not None:
        c = h["comparison"]
        return (f"{_when(h['at'], h['days'])}{stop}: {_result(c)}; "
                + _verdict(c.verdict, c.diff, c.diff_pts))
    return (f"{_when(h['at'], h['days'])}{stop}: bot "
            f"{_money(h['total_profit'])}; {h['note']}")


def render_show(bot: dict, hist: list[dict]) -> str:
    lines = [_head(bot)]
    extra = []
    if bot.get("range_low") is not None:
        extra.append(f"range {bot['range_low']:,.2f} to {bot['range_high']:,.2f}")
    if bot.get("grids") is not None:
        extra.append(f"{bot['grids']} grids")
    if extra:
        lines.append("   " + ", ".join(extra))
    lines.append("")
    if not hist:
        lines.append(f"no update yet (tb bots update {bot['id']} --profit X)")
    lines += [reading_line(h) for h in hist]
    lines += ["", FOOTER]
    return "\n".join(lines)
