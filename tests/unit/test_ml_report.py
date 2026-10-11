"""The evaluation report (spec section 6)."""
import re
from datetime import datetime, timezone

from ml.report import render_report, verdict

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)
NAMES = ("base_rate_v1", "ema_cross_v1", "rsi_v1", "macd_v1", "logreg_v2", "xgb_v1")


def test_verdict_phrasings():
    assert verdict(0.008, 0.003, 0.013) == (
        "skill +0.8% (95% CI +0.3% to +1.3%): better than the base rate on unseen data")
    assert verdict(0.001, -0.004, 0.006) == (
        "skill +0.1% (95% CI -0.4% to +0.6%): not distinguishable from the base rate")
    assert verdict(-0.005, -0.009, -0.001) == (
        "skill -0.5% (95% CI -0.9% to -0.1%): worse than the base rate")


def _model(skill):
    return {"log_loss": 1.05, "brier": 0.62, "accuracy": 0.41, "ece": 0.01,
            "skill": skill, "skill_lo": skill - 0.004, "skill_hi": skill + 0.004,
            "reliability": [],
            "per_fold": [{"fold": 0, "test_start": "2020-01-01T01:00:00+00:00",
                          "n": 100, "skill": skill}],
            "per_symbol": {"BTCUSDT": skill, "ETHUSDT": -skill}}


def _run(kind="walk_forward", holdout_count=0):
    return {
        "run_id": 7, "kind": kind, "horizon": 4, "holdout_count": holdout_count,
        "created_at": datetime(2026, 10, 7, tzinfo=timezone.utc),
        "data_end": datetime(2026, 10, 7, 8, tzinfo=timezone.utc),
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "metrics": {"horizon": 4, "n_test": 1000,
                    "class_shares": {"down": .3, "flat": .4, "up": .3},
                    "models": {n: _model(0.0 if n == "base_rate_v1" else 0.01)
                               for n in NAMES},
                    "folds": [{"number": 0, "test_start": "2020-01-01T00:00:00+00:00",
                               "test_end": "2020-07-01T00:00:00+00:00",
                               "n_fit": 1, "n_cal": 1, "n_test": 1}]},
    }


def test_report_contents_and_no_advice_words():
    text = render_report(_run())
    assert not ADVICE.search(text), ADVICE.search(text)
    for name in NAMES:
        assert name in text
    assert "next 4h" in text and "flat 40%" in text
    assert "untuned" in text
    assert "better than the base rate on unseen data" in text


def test_holdout_warning_from_second_run():
    assert "Warning" not in render_report(_run("holdout", 1))
    assert ("Warning: the holdout has now been evaluated 2 times; this result "
            "is no longer an unbiased estimate.") in render_report(_run("holdout", 2))
    assert "Warning" not in render_report(_run("walk_forward", 5))


def test_models_in_fixed_order_whatever_the_stored_order():
    # PostgreSQL jsonb returns object keys sorted by length, not insertion order.
    run = _run()
    models = run["metrics"]["models"]
    run["metrics"]["models"] = {k: models[k] for k in sorted(models, key=len)}
    text = render_report(run)
    table = [l.split()[0] for l in text.splitlines() if l.split() and l.split()[0] in NAMES]
    assert table[:6] == list(NAMES)


def test_report_names_the_target_classes_in_order():
    run = _run()
    run["target"] = "dir2"
    # jsonb returns keys sorted by length: "up" before "down"
    run["metrics"]["class_shares"] = {"up": .52, "down": .48}
    text = render_report(run)
    assert "target dir2" in text
    assert "Outcomes in the test periods: down 48%, up 52%" in text


def test_holdout_count_says_all_targets():
    text = render_report(_run("holdout", 1))
    assert "Holdout evaluations for this horizon so far (all targets): 1" in text
