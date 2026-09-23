from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Nothing is hardcoded, so moving to a VPS
    means changing .env and nothing else."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    test_database_url: str = ""
    symbols: Annotated[list[str], NoDecode] = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
    ]
    binance_base_url: str = "https://api.binance.com"
    binance_ws_url: str = "wss://stream.binance.com:9443"
    binance_data_url: str = "https://data.binance.vision"
    data_cache_dir: Path = Path("./data_cache")
    log_level: str = "INFO"

    @field_validator("symbols", mode="before")
    @classmethod
    def _split_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return [str(s).strip().upper() for s in v]


@lru_cache
def get_settings() -> Settings:
    return Settings()
