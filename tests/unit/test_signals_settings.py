import pytest

from signals.settings import RiskSettings, validate


def test_defaults():
    s = RiskSettings()
    assert (s.profile, s.wild_threshold, s.capital_usdt, s.max_coin_share, s.fee) == \
        ("defensive", 0.25, None, 0.25, 0.00075)
    validate(s)


@pytest.mark.parametrize("change", [
    {"profile": "bold"}, {"wild_threshold": 0.04}, {"wild_threshold": 0.96},
    {"capital_usdt": -1.0}, {"max_coin_share": 0.0}, {"max_coin_share": 1.01},
    {"fee": -0.001}, {"fee": 0.02}, {"wild_threshold": float("nan")},
])
def test_bad_values_raise(change):
    with pytest.raises(ValueError):
        validate(RiskSettings(**change))


def test_edges_are_allowed():
    validate(RiskSettings(profile="return", wild_threshold=0.05, capital_usdt=0.0,
                          max_coin_share=1.0, fee=0.01))
    validate(RiskSettings(wild_threshold=0.95, max_coin_share=0.01, fee=0.0))
