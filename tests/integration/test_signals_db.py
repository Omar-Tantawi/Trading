"""Risk settings and stored signals against the test database."""
import pytest

from signals.settings import RiskSettings, load_settings, save_settings

pytestmark = pytest.mark.db


def test_no_row_gives_defaults(db_conn):
    assert load_settings(db_conn) == RiskSettings()


def test_settings_round_trip(db_conn):
    s = RiskSettings(profile="return", wild_threshold=0.3, capital_usdt=2000.0,
                     max_coin_share=0.5, fee=0.001)
    save_settings(db_conn, s)
    save_settings(db_conn, s)          # still one row
    assert load_settings(db_conn) == s
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM risk_settings")
        assert cur.fetchone()[0] == 1


def test_save_refuses_bad_settings(db_conn):
    with pytest.raises(ValueError):
        save_settings(db_conn, RiskSettings(wild_threshold=2.0))
    assert load_settings(db_conn) == RiskSettings()
