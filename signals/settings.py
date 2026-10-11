"""The user's risk settings (spec section 3): one row, defaults when absent."""
import math
from dataclasses import asdict, dataclass

PROFILES = ("defensive", "return")


@dataclass(frozen=True)
class RiskSettings:
    profile: str = "defensive"
    wild_threshold: float = 0.25
    capital_usdt: float | None = None
    max_coin_share: float = 0.25
    fee: float = 0.00075

    def to_dict(self) -> dict:
        return asdict(self)


def _within(name: str, value: float, lo: float, hi: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value)
            and lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}, got {value}")


def validate(s: RiskSettings) -> None:
    if s.profile not in PROFILES:
        raise ValueError(f"profile must be one of {', '.join(PROFILES)}, got {s.profile!r}")
    _within("wild_threshold", s.wild_threshold, 0.05, 0.95)
    if s.capital_usdt is not None:
        _within("capital_usdt", s.capital_usdt, 0.0, math.inf)
    _within("max_coin_share", s.max_coin_share, 0.01, 1.0)
    _within("fee", s.fee, 0.0, 0.01)


def load_settings(conn) -> RiskSettings:
    with conn.cursor() as cur:
        cur.execute("SELECT profile, wild_threshold, capital_usdt, max_coin_share, fee "
                    "FROM risk_settings WHERE id = 1")
        row = cur.fetchone()
    return RiskSettings(*row) if row else RiskSettings()


def save_settings(conn, s: RiskSettings) -> None:
    validate(s)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO risk_settings (id, profile, wild_threshold, capital_usdt, "
            "max_coin_share, fee) VALUES (1, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET profile = EXCLUDED.profile, "
            "wild_threshold = EXCLUDED.wild_threshold, "
            "capital_usdt = EXCLUDED.capital_usdt, "
            "max_coin_share = EXCLUDED.max_coin_share, fee = EXCLUDED.fee, "
            "updated_at = now()",
            (s.profile, s.wild_threshold, s.capital_usdt, s.max_coin_share, s.fee))
    conn.commit()
