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
    assert text.count("grid_vol3_30 vs grid_random_30:") == 3
    assert "All coins" in text and "Not advice." in text
    assert not ADVICE.search(text)


def test_report_has_averaged_table_and_disclosures():
    coins = {s: {n: run_strategy(_data(days=30, seed=i), n, 0.001) for n in STRATEGIES}
             for i, s in enumerate(("BTCUSDT", "ETHUSDT"))}
    text = render(7, 0.001, 0.0005, coins)
    assert "All coins, equal weight" in text
    avg = text.split("All coins, equal weight")[1]
    assert "  grid_always " in avg
    assert "full 1440 minutes" in text and "random pause days are drawn over the whole period" in text
