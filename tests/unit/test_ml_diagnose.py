"""Splitting skill into size (move vs flat) and direction (up vs down)."""
import re

import numpy as np
import pandas as pd
import pytest

from ml.diagnose import render_diagnosis, split_skill

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)


def _pred(n=4000, seed=0, size_info=False, direction_info=False):
    rng = np.random.default_rng(seed)
    label = rng.choice([0, 1, 2], size=n, p=[.2, .6, .2])
    times = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = np.tile([.2, .6, .2], (n, 1))
    model = base.copy()
    moved = label != 1
    if size_info:      # more weight on "a move" exactly when one happens
        model = np.where(moved[:, None], [.3, .4, .3], [.15, .7, .15])
    if direction_info:  # the right side of a move gets more weight
        up = label == 2
        model = model.copy()
        u = moved & up
        model[u] = np.column_stack([model[u][:, 0] * .5, model[u][:, 1],
                                    model[u][:, 2] + model[u][:, 0] * .5])
        dn = moved & ~up
        model[dn] = np.column_stack([model[dn][:, 0] + model[dn][:, 2] * .5,
                                     model[dn][:, 1], model[dn][:, 2] * .5])
    rows = []
    for name, p in (("base_rate_v1", base), ("xgb_v1", model)):
        rows.append(pd.DataFrame({"model": name, "symbol": "BTCUSDT", "open_time": times,
                                  "fold": 0, "label": label, "p_down": p[:, 0],
                                  "p_flat": p[:, 1], "p_up": p[:, 2]}))
    return pd.concat(rows, ignore_index=True)


def test_size_only_information():
    d = split_skill(_pred(size_info=True))["xgb_v1"]
    assert d["size"]["skill_lo"] > 0
    assert d["direction"]["skill"] == pytest.approx(0.0, abs=1e-9)
    assert d["direction"]["n"] == int((_pred()["label"] != 1).sum() / 2)


def test_direction_only_information():
    d = split_skill(_pred(direction_info=True))["xgb_v1"]
    assert d["direction"]["skill_lo"] > 0
    assert d["direction"]["accuracy"] > d["direction"]["base_accuracy"]
    assert d["size"]["skill"] == pytest.approx(0.0, abs=1e-9)


def test_base_rate_not_reported_and_render_has_no_advice():
    out = split_skill(_pred(size_info=True, direction_info=True))
    assert set(out) == {"xgb_v1"}
    text = render_diagnosis(7, 4, out)
    assert "Run 7" in text and "next 4h" in text
    assert "size (move vs flat)" in text and "direction (up vs down" in text
    assert not ADVICE.search(text)
