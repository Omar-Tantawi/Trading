"""The evaluation report (spec section 6): measurement, never advice."""

from ml.targets import get_target

BASE = "base_rate_v1"
MODEL_ORDER = ("base_rate_v1", "ema_cross_v1", "rsi_v1", "macd_v1",
               "logreg_v1", "logreg_v2", "xgb_v1")


def _ordered(models: dict) -> list[tuple[str, dict]]:
    """Models in report order. Stored metrics come back from a jsonb column,
    which does not keep key order."""
    rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(models.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0]))


def _pct(v: float) -> str:
    return f"{v:+.1%}"


def verdict(skill: float, lo: float, hi: float) -> str:
    ci = f"skill {_pct(skill)} (95% CI {_pct(lo)} to {_pct(hi)})"
    if lo > 0:
        return f"{ci}: better than the base rate on unseen data"
    if hi < 0:
        return f"{ci}: worse than the base rate"
    return f"{ci}: not distinguishable from the base rate"


def render_report(run: dict) -> str:
    m = run["metrics"]
    h = run["horizon"]
    models = _ordered(m["models"])
    lines = [
        f"Run {run['run_id']} ({run['kind'].replace('_', '-')}), "
        f"target {run.get('target', 'move3')}, next {h}h, "
        f"{', '.join(run['symbols'])}; data to "
        f"{run['data_end']:%Y-%m-%d %H:%M} UTC; {m['n_test']:,} test rows",
    ]
    if run["kind"] == "holdout":
        n = run["holdout_count"]
        # Counted across targets: once any target has looked at the holdout
        # period, it is no longer unseen.
        lines.append(f"Holdout evaluations for this horizon so far (all targets): {n}")
        if n >= 2:
            lines.append(f"Warning: the holdout has now been evaluated {n} times; "
                         "this result is no longer an unbiased estimate.")
    shares = m["class_shares"]
    target = get_target(run.get("target", "move3"))
    lines.append("Outcomes in the test periods: " + ", ".join(
        f"{c} {shares[c]:.0%}" for c in target.classes if c in shares))
    lines.append("")
    lines.append(f"{'model':<14}{'log loss':>9}{'skill':>8}{'95% CI':>18}"
                 f"{'Brier':>8}{'acc':>7}{'ECE':>7}")
    for name, s in models:
        ci = "-" if name == BASE else f"{_pct(s['skill_lo'])}..{_pct(s['skill_hi'])}"
        lines.append(f"{name:<14}{s['log_loss']:>9.4f}{_pct(s['skill']):>8}"
                     f"{ci:>18}{s['brier']:>8.4f}{s['accuracy']:>7.1%}{s['ece']:>7.3f}")
    lines.append("")
    lines.append("Skill per test period (does it hold up over time?):")
    for name, s in models:
        if name == BASE:
            continue
        lines.append(f"  {name:<14}" + " ".join(
            f"{f['test_start'][:7]}:{_pct(f['skill'])}" for f in s["per_fold"]))
    lines.append("Skill per symbol:")
    for name, s in models:
        if name == BASE:
            continue
        lines.append(f"  {name:<14}" + " ".join(
            f"{sym} {_pct(v)}" for sym, v in s["per_symbol"].items()))
    lines.append("")
    for name, s in models:
        if name != BASE:
            lines.append(f"{name}: {verdict(s['skill'], s['skill_lo'], s['skill_hi'])}")
    lines.append("")
    lines.append("Models are untuned (fixed settings, spec section 5.2). Skill is "
                 "measured against always predicting the training base rate.")
    return "\n".join(lines)
