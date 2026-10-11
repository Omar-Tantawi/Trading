import re

from backtest.engine import STRATEGIES, run_strategy
from backtest.report import render
from tests.unit.test_backtest_engine import _data

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)


def test_report_rows_verdicts_and_no_advice():
    coins = {s: {n: run_strategy(_data(days=30, seed=i), n, 0.001) for n in STRATEGIES}
             for i, s in enumerate(("BTCUSDT", "ETHUSDT"))}
    text = render(7, 0.001, 0.0005, coins)
    for name in STRATEGIES:
        assert f"  {name} " in text
    assert text.count("grid_vol3_30 vs grid_always:") == 3       # 2 coins + pooled
    assert "All coins" in text and "Not advice." in text
    assert not ADVICE.search(text)
