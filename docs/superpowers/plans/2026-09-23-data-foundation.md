# Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verified, gap-checked Binance market-data warehouse (full 1-minute history plus a live collector) that every later sub-project of the AI Trading Buddy reads from.

**Architecture:** Binance bulk archives and REST fill history; a WebSocket collector keeps it current. Everything lands in `candles_1m` in PostgreSQL + TimescaleDB, from which 5m/15m/1h/4h/1d are derived as continuous aggregates, so higher timeframes can never disagree with the 1m truth. A separate quality engine audits the result and can block model training.

**Tech Stack:** Python 3.13.3, psycopg 3, httpx, websockets, pydantic-settings, Typer, pandas (tests only), pytest, Docker + timescale/timescaledb (PostgreSQL 16).

**Spec:** `docs/superpowers/specs/2026-09-23-data-foundation-design.md`

## Global Constraints

- **Python 3.13.3** via `py -3.13`; the machine's Python 3.9 is never used.
- **Docker Desktop is installed per-user** at `C:\Users\ASUS\AppData\Local\Programs\DockerDesktop`; `docker` is already on PATH.
- **All timestamps are timezone-aware UTC** (`datetime.timezone.utc`). Naive datetimes are a bug; reject them at boundaries.
- **All prices and volumes use `decimal.Decimal`, never `float`.** Parse from the raw string. A float round-trip of `0.00000001` loses data and corrupts every downstream feature.
- **Every write is idempotent**, keyed on `(symbol, open_time)`. Any component may be killed and restarted at any time.
- **Source precedence is `archive` > `rest` > `ws`.** A lower-precedence source must never overwrite a higher one.
- **No secrets in the repo.** Configuration comes from `.env` (git-ignored); `.env.example` is committed. Sub-project 1 needs no Binance API key at all.
- **Tests run offline by default.** Anything touching the network is marked `@pytest.mark.network` and deselected in the default run.
- **Commit after every task.** Conventional commit messages (`feat:`, `test:`, `chore:`).
- Symbols for this sub-project: `BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT`.

## Review Focus

Five things the spec implies, that a naive implementation gets wrong, and that would bite a real user. Each has a test pinned to the task that owns the code:

1. **Archive timestamp units changed.** Binance archives written from 2025 onward use **microseconds**; older files use milliseconds. Parsing both as ms puts candles in the year 57000. Test in Task 4.
2. **Decimal precision loss.** Parsing prices via `float` silently mangles satoshi-level values. Test in Task 4.
3. **Months before a symbol was listed return HTTP 404.** SOLUSDT has no 2017 archive. A 404 on a pre-listing month is normal and must be skipped, not retried forever or recorded as failure. Test in Task 5.
4. **Genuine Binance downtime looks identical to data loss.** The exchange has had multi-hour outages; those minutes do not exist anywhere. The quality engine must report them as gaps but must not fail forever on unfixable history, or `assert_trainable` becomes permanently red and useless. Test in Task 8.
5. **Rate limiting (HTTP 429/418).** Backfilling four symbols hammers the REST API; ignoring `Retry-After` gets the IP banned for hours. Test in Task 3.

---

## File Structure

```text
docker-compose.yml           TimescaleDB service + named volume
pyproject.toml               deps, pytest config, `tb` entry point
.env.example                 committed template
.gitignore
data/
  __init__.py
  config.py                  Settings (pydantic-settings)
  cli.py                     Typer app: db/symbols/backfill/live/quality/status
  storage/
    __init__.py
    db.py                    connection pool, migration runner
    migrations/001_initial.sql
    repository.py            upsert/query functions
  collectors/
    __init__.py
    binance_rest.py          HTTP client: retry, backoff, 429/418 handling
    binance_ws.py            WebSocket client: reconnect, message parsing
    archive.py               URL building, checksum verify, CSV parsing
    backfill.py              orchestration: archives -> REST tail, resumable
    live.py                  live collector loop
  quality/
    __init__.py
    checks.py                the rules + assert_trainable
    report.py                Report dataclass + text rendering
tests/
  conftest.py                db fixtures
  fixtures/                  sample CSV/zip/json, committed
  unit/                      offline
  integration/               marked `network` or `db`
```

---

### Task 1: Project scaffold, config, and the TimescaleDB container

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `docker-compose.yml`
- Create: `data/__init__.py`, `data/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `data.config.Settings` with fields `database_url: str`, `symbols: list[str]`, `binance_base_url: str`, `binance_ws_url: str`, `binance_data_url: str`, `data_cache_dir: Path`, `log_level: str`; and `get_settings() -> Settings` (cached).

- [ ] **Step 1: Extend the git-ignore file**

`.gitignore` already exists and already ignores `.superpowers/`. **Keep that
line** and add the rest, so the file reads:
```gitignore
.superpowers/
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
data_cache/
*.egg-info/
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "trading-buddy"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "httpx>=0.27",
    "websockets>=13.0",
    "psycopg[binary,pool]>=3.2",
    "pydantic-settings>=2.4",
    "typer>=0.12",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pandas>=2.2",
    "respx>=0.21",
]

[project.scripts]
tb = "data.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["data*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "network: hits the real Binance API (deselected by default)",
    "db: needs the TimescaleDB container",
]
addopts = "-m 'not network'"
```

- [ ] **Step 3: Create `.env.example`**

```dotenv
DATABASE_URL=postgresql://tb:tb_local_dev@localhost:5432/trading_buddy
TEST_DATABASE_URL=postgresql://tb:tb_local_dev@localhost:5432/trading_buddy_test
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
BINANCE_BASE_URL=https://api.binance.com
BINANCE_WS_URL=wss://stream.binance.com:9443
BINANCE_DATA_URL=https://data.binance.vision
DATA_CACHE_DIR=./data_cache
LOG_LEVEL=INFO
```

- [ ] **Step 4: Create `docker-compose.yml`**

```yaml
services:
  db:
    image: timescale/timescaledb:latest-pg16
    container_name: tb-timescaledb
    restart: unless-stopped
    environment:
      POSTGRES_USER: tb
      POSTGRES_PASSWORD: tb_local_dev
      POSTGRES_DB: trading_buddy
    ports:
      - "5432:5432"
    volumes:
      - tb_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tb -d trading_buddy"]
      interval: 5s
      timeout: 5s
      retries: 20
    command:
      - postgres
      - -c
      - shared_preload_libraries=timescaledb
      - -c
      - max_wal_size=4GB
      - -c
      - shared_buffers=1GB
      - -c
      - work_mem=64MB

volumes:
  tb_pgdata:
```

- [ ] **Step 5: Start the database and confirm TimescaleDB loads**

Run:
```bash
docker compose up -d
```
Then:
```bash
docker exec tb-timescaledb psql -U tb -d trading_buddy -c "CREATE EXTENSION IF NOT EXISTS timescaledb; SELECT extname, extversion FROM pg_extension WHERE extname='timescaledb';"
```
Expected: one row naming `timescaledb` and a version. Record that version and the exact image digest in the commit message, then pin the image tag in `docker-compose.yml` to the version reported (for example `timescale/timescaledb:2.17.2-pg16`) so a future `docker compose pull` cannot silently change the database.

- [ ] **Step 6: Create the test database**

Run:
```bash
docker exec tb-timescaledb psql -U tb -d trading_buddy -c "CREATE DATABASE trading_buddy_test OWNER tb;"
```

- [ ] **Step 7: Create the virtual environment and install**

Run:
```bash
py -3.13 -m venv .venv
```
Then (Windows PowerShell):
```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
```
Then:
```bash
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 8: Write the failing config test**

`tests/unit/test_config.py`:
```python
from pathlib import Path

from data.config import Settings


def test_settings_parses_symbols_and_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT, ethusdt ,SOLUSDT")
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))

    s = Settings()

    assert s.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert isinstance(s.data_cache_dir, Path)
    assert s.binance_base_url == "https://api.binance.com"


def test_settings_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TB_IGNORE_DOTENV", "1")
    try:
        Settings(_env_file=None)
    except Exception as exc:
        assert "database_url" in str(exc).lower()
    else:
        raise AssertionError("expected a validation error")
```

- [ ] **Step 9: Run the test and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'data.config'`.

- [ ] **Step 10: Implement the config module**

`data/__init__.py`: empty file.

`data/config.py`:
```python
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Nothing is hardcoded, so moving to a VPS
    means changing .env and nothing else."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    test_database_url: str = ""
    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
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
```

- [ ] **Step 11: Run the tests and watch them pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -v
```
Expected: 2 passed.

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml .gitignore .env.example docker-compose.yml data tests
git commit -m "feat: project scaffold, settings, and TimescaleDB container"
```

---

### Task 2: Database schema, migrations, and the storage layer

**Files:**
- Create: `data/storage/__init__.py`, `data/storage/db.py`, `data/storage/migrations/001_initial.sql`, `data/storage/repository.py`
- Create: `tests/conftest.py`, `tests/integration/test_schema.py`, `tests/integration/test_repository.py`

**Interfaces:**
- Consumes: `data.config.get_settings`.
- Produces:
  - `data.storage.db.connect(dsn: str | None = None) -> psycopg.Connection`
  - `data.storage.db.run_migrations(dsn: str | None = None) -> list[str]` (returns applied filenames)
  - `data.storage.repository.Candle` — dataclass: `symbol: str`, `open_time: datetime`, `close_time: datetime`, `open/high/low/close: Decimal`, `volume: Decimal`, `quote_volume: Decimal`, `trade_count: int`, `taker_buy_base: Decimal`, `taker_buy_quote: Decimal`, `source: str`
  - `upsert_candles(conn, candles: Iterable[Candle]) -> int`
  - `last_candle_time(conn, symbol: str) -> datetime | None`
  - `get_candles(conn, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[dict]`
  - `upsert_symbol(conn, **fields) -> None`
  - `start_run(conn, component, symbol, period_start, period_end) -> int`, `finish_run(conn, run_id, status, rows_written, error=None) -> None`, `completed_periods(conn, component, symbol) -> set[tuple[datetime, datetime]]`

- [ ] **Step 1: Write the migration SQL**

`data/storage/migrations/001_initial.sql`:
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS symbols (
    symbol        text PRIMARY KEY,
    base_asset    text,
    quote_asset   text,
    tick_size     numeric,
    step_size     numeric,
    min_notional  numeric,
    listed_at     timestamptz,
    status        text,
    is_active     boolean DEFAULT true,
    updated_at    timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS candles_1m (
    symbol          text NOT NULL REFERENCES symbols(symbol),
    open_time       timestamptz NOT NULL,
    close_time      timestamptz NOT NULL,
    open            numeric(20,8) NOT NULL,
    high            numeric(20,8) NOT NULL,
    low             numeric(20,8) NOT NULL,
    close           numeric(20,8) NOT NULL,
    volume          numeric(30,8) NOT NULL,
    quote_volume    numeric(30,8) NOT NULL,
    trade_count     integer NOT NULL,
    taker_buy_base  numeric(30,8) NOT NULL,
    taker_buy_quote numeric(30,8) NOT NULL,
    source          text NOT NULL,
    inserted_at     timestamptz DEFAULT now(),
    PRIMARY KEY (symbol, open_time)
);

SELECT create_hypertable('candles_1m', 'open_time',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS book_ticker (
    symbol     text NOT NULL,
    ts         timestamptz NOT NULL,
    bid_price  numeric(20,8) NOT NULL,
    bid_qty    numeric(30,8) NOT NULL,
    ask_price  numeric(20,8) NOT NULL,
    ask_qty    numeric(30,8) NOT NULL,
    spread     numeric(20,8) GENERATED ALWAYS AS (ask_price - bid_price) STORED,
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('book_ticker', 'ts',
                         chunk_time_interval => INTERVAL '1 day',
                         if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id            bigserial PRIMARY KEY,
    component     text NOT NULL,
    symbol        text,
    period_start  timestamptz,
    period_end    timestamptz,
    status        text NOT NULL,
    rows_written  bigint DEFAULT 0,
    error         text,
    started_at    timestamptz DEFAULT now(),
    finished_at   timestamptz
);

CREATE INDEX IF NOT EXISTS ingestion_runs_lookup
    ON ingestion_runs (component, symbol, status, period_start);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    id              bigserial PRIMARY KEY,
    symbol          text NOT NULL,
    timeframe       text NOT NULL,
    checked_from    timestamptz,
    checked_to      timestamptz,
    total_candles   bigint,
    duplicates      bigint,
    invalid         bigint,
    missing         bigint,
    completeness_pct numeric(9,6),
    verdict         text NOT NULL,
    details         jsonb,
    created_at      timestamptz DEFAULT now()
);

-- Source precedence: archive (2) > rest (1) > ws (0).
CREATE OR REPLACE FUNCTION source_priority(s text) RETURNS int AS $$
    SELECT CASE s WHEN 'archive' THEN 2 WHEN 'rest' THEN 1 ELSE 0 END;
$$ LANGUAGE sql IMMUTABLE;
```

- [ ] **Step 2: Write the continuous-aggregate part of the migration**

Append to `001_initial.sql`. Note `first()`/`last()` need the ordering column, and continuous aggregates cannot be created inside a transaction that already created the hypertable in some Timescale versions — the migration runner in Step 5 therefore runs each file with autocommit.

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_5m
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket(INTERVAL '5 minutes', open_time) AS open_time,
       first(open, open_time)  AS open,
       max(high)               AS high,
       min(low)                AS low,
       last(close, open_time)  AS close,
       sum(volume)             AS volume,
       sum(quote_volume)       AS quote_volume,
       sum(trade_count)        AS trade_count,
       sum(taker_buy_base)     AS taker_buy_base,
       sum(taker_buy_quote)    AS taker_buy_quote
FROM candles_1m
GROUP BY symbol, 2
WITH NO DATA;
```

Repeat the identical block for `candles_15m` (`INTERVAL '15 minutes'`), `candles_1h` (`INTERVAL '1 hour'`), `candles_4h` (`INTERVAL '4 hours'`) and `candles_1d` (`INTERVAL '1 day'`), changing only the view name and the bucket interval.

Then the refresh policies:
```sql
SELECT add_continuous_aggregate_policy('candles_5m',
    start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_15m',
    start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '15 minutes', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_1h',
    start_offset => INTERVAL '30 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_4h',
    start_offset => INTERVAL '90 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_1d',
    start_offset => INTERVAL '365 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);

ALTER TABLE candles_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('candles_1m', INTERVAL '7 days', if_not_exists => TRUE);
```

- [ ] **Step 3: Write the failing schema test**

`tests/integration/test_schema.py`:
```python
import pytest

pytestmark = pytest.mark.db


def test_migrations_create_hypertables(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT hypertable_name FROM timescaledb_information.hypertables"
        )
        names = {r[0] for r in cur.fetchall()}
    assert {"candles_1m", "book_ticker"} <= names


def test_migrations_create_continuous_aggregates(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT view_name FROM timescaledb_information.continuous_aggregates"
        )
        views = {r[0] for r in cur.fetchall()}
    assert {"candles_5m", "candles_15m", "candles_1h", "candles_4h", "candles_1d"} <= views


def test_migrations_are_idempotent(db_conn):
    from data.storage.db import run_migrations

    applied_again = run_migrations()
    assert applied_again == []
```

- [ ] **Step 4: Write the conftest fixtures**

`tests/conftest.py`:
```python
import os

import pytest

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://tb:tb_local_dev@localhost:5432/trading_buddy_test",
)


@pytest.fixture(scope="session")
def migrated_db():
    """Apply migrations once against the test database."""
    os.environ["DATABASE_URL"] = TEST_DSN
    from data.config import get_settings
    from data.storage.db import run_migrations

    get_settings.cache_clear()
    run_migrations(TEST_DSN)
    return TEST_DSN


@pytest.fixture
def db_conn(migrated_db):
    """A connection to a database emptied before each test.

    DELETE rather than TRUNCATE: TimescaleDB restricts TRUNCATE on a
    hypertable that has continuous aggregates attached.
    """
    from data.storage.db import connect

    conn = connect(migrated_db)
    try:
        with conn.cursor() as cur:
            for table in ("candles_1m", "book_ticker", "ingestion_runs",
                          "data_quality_reports", "symbols"):
                cur.execute(f"DELETE FROM {table}")
        conn.commit()
        yield conn
    finally:
        conn.close()


@pytest.fixture
def autocommit_conn(migrated_db):
    """Some TimescaleDB commands (CALL refresh_continuous_aggregate) refuse
    to run inside a transaction block."""
    import psycopg

    conn = psycopg.connect(migrated_db, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 5: Run the schema test and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_schema.py -v
```
Expected: FAIL — `No module named 'data.storage.db'`.

- [ ] **Step 6: Implement the migration runner**

`data/storage/__init__.py`: empty file.

`data/storage/db.py`:
```python
from pathlib import Path

import psycopg

from data.config import get_settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(dsn: str | None = None) -> psycopg.Connection:
    dsn = dsn or get_settings().database_url
    return psycopg.connect(dsn)


def _ensure_migrations_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename    text PRIMARY KEY,
                applied_at  timestamptz DEFAULT now()
            )
            """
        )


def run_migrations(dsn: str | None = None) -> list[str]:
    """Apply any .sql files not yet recorded. Returns the files applied.

    Autocommit is required: TimescaleDB refuses to create continuous
    aggregates inside an explicit transaction block.
    """
    dsn = dsn or get_settings().database_url
    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        _ensure_migrations_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM schema_migrations")
            done = {r[0] for r in cur.fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (path.name,),
                )
            applied.append(path.name)
    return applied
```

- [ ] **Step 7: Run the schema test and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_schema.py -v -m db
```
Expected: 3 passed. If a continuous-aggregate statement errors, read the message — Timescale is explicit about what it refuses and why.

- [ ] **Step 8: Write the failing repository test**

`tests/integration/test_repository.py`:
```python
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.storage.repository import (
    Candle,
    last_candle_time,
    upsert_candles,
    upsert_symbol,
)

pytestmark = pytest.mark.db

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_candle(minute: int, close: str = "42000.12345678", source: str = "archive"):
    open_time = T0 + timedelta(minutes=minute)
    return Candle(
        symbol="BTCUSDT",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=Decimal("42000.00000001"),
        high=Decimal("42100.5"),
        low=Decimal("41900.25"),
        close=Decimal(close),
        volume=Decimal("1.23456789"),
        quote_volume=Decimal("51840.5"),
        trade_count=42,
        taker_buy_base=Decimal("0.6"),
        taker_buy_quote=Decimal("25000.1"),
        source=source,
    )


def test_upsert_is_idempotent_and_preserves_precision(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [make_candle(i) for i in range(10)]

    assert upsert_candles(db_conn, rows) == 10
    upsert_candles(db_conn, rows)

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*), min(open) FROM candles_1m")
        count, min_open = cur.fetchone()
    assert count == 10
    assert min_open == Decimal("42000.00000001")


def test_lower_precedence_source_does_not_overwrite(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [make_candle(0, close="1.0", source="archive")])
    upsert_candles(db_conn, [make_candle(0, close="999.0", source="ws")])

    with db_conn.cursor() as cur:
        cur.execute("SELECT close, source FROM candles_1m")
        close, source = cur.fetchone()
    assert close == Decimal("1.00000000")
    assert source == "archive"


def test_higher_precedence_source_does_overwrite(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [make_candle(0, close="1.0", source="ws")])
    upsert_candles(db_conn, [make_candle(0, close="999.0", source="archive")])

    with db_conn.cursor() as cur:
        cur.execute("SELECT close, source FROM candles_1m")
        close, source = cur.fetchone()
    assert close == Decimal("999.00000000")
    assert source == "archive"


def test_last_candle_time(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [make_candle(i) for i in range(5)])
    assert last_candle_time(db_conn, "BTCUSDT") == T0 + timedelta(minutes=4)
    assert last_candle_time(db_conn, "ETHUSDT") is None


def test_rejects_naive_datetime(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    bad = make_candle(0)
    bad.open_time = bad.open_time.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        upsert_candles(db_conn, [bad])
```

- [ ] **Step 9: Run it and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_repository.py -v -m db
```
Expected: FAIL — `No module named 'data.storage.repository'`.

- [ ] **Step 10: Implement the repository**

`data/storage/repository.py`:
```python
from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from typing import Iterable

import psycopg

CANDLE_COLUMNS = (
    "symbol", "open_time", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "trade_count", "taker_buy_base",
    "taker_buy_quote", "source",
)


@dataclass
class Candle:
    symbol: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trade_count: int
    taker_buy_base: Decimal
    taker_buy_quote: Decimal
    source: str

    def as_row(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))


def _check_aware(dt: datetime, label: str) -> None:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{label} must be timezone-aware UTC, got {dt!r}")


def upsert_symbol(conn: psycopg.Connection, **fields_) -> None:
    cols = list(fields_)
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "symbol")
    sql = f"INSERT INTO symbols ({', '.join(cols)}) VALUES ({placeholders})"
    sql += f" ON CONFLICT (symbol) DO UPDATE SET {updates}, updated_at = now()" if updates \
        else " ON CONFLICT (symbol) DO NOTHING"
    with conn.cursor() as cur:
        cur.execute(sql, tuple(fields_.values()))
    conn.commit()


def upsert_candles(conn: psycopg.Connection, candles: Iterable[Candle]) -> int:
    rows = list(candles)
    if not rows:
        return 0
    for c in rows:
        _check_aware(c.open_time, "open_time")
        _check_aware(c.close_time, "close_time")

    cols = ", ".join(CANDLE_COLUMNS)
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in CANDLE_COLUMNS
        if c not in ("symbol", "open_time")
    )
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS staging_candles "
            "(LIKE candles_1m INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        cur.execute("TRUNCATE staging_candles")
        with cur.copy(f"COPY staging_candles ({cols}) FROM STDIN") as copy:
            for c in rows:
                copy.write_row(c.as_row())
        cur.execute(
            f"""
            INSERT INTO candles_1m ({cols})
            SELECT {cols} FROM staging_candles
            ON CONFLICT (symbol, open_time) DO UPDATE SET {updates}
            WHERE source_priority(EXCLUDED.source)
                  >= source_priority(candles_1m.source)
            """
        )
    conn.commit()
    return len(rows)


def upsert_book_ticker(conn: psycopg.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        with cur.copy(
            "COPY book_ticker (symbol, ts, bid_price, bid_qty, ask_price, ask_qty) "
            "FROM STDIN"
        ) as copy:
            for r in rows:
                copy.write_row(r)
    conn.commit()
    return len(rows)


def last_candle_time(conn: psycopg.Connection, symbol: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT max(open_time) FROM candles_1m WHERE symbol = %s", (symbol,)
        )
        return cur.fetchone()[0]


def get_candles(conn, symbol: str, timeframe: str, start: datetime,
                end: datetime) -> list[dict]:
    table = "candles_1m" if timeframe == "1m" else f"candles_{timeframe}"
    if table not in {"candles_1m", "candles_5m", "candles_15m", "candles_1h",
                     "candles_4h", "candles_1d"}:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM {table} WHERE symbol = %s AND open_time >= %s "
            f"AND open_time < %s ORDER BY open_time",
            (symbol, start, end),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def start_run(conn, component: str, symbol: str | None,
              period_start: datetime | None, period_end: datetime | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingestion_runs (component, symbol, period_start, "
            "period_end, status) VALUES (%s, %s, %s, %s, 'running') RETURNING id",
            (component, symbol, period_start, period_end),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_run(conn, run_id: int, status: str, rows_written: int = 0,
               error: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingestion_runs SET status = %s, rows_written = %s, "
            "error = %s, finished_at = now() WHERE id = %s",
            (status, rows_written, error, run_id),
        )
    conn.commit()


def completed_periods(conn, component: str, symbol: str) -> set[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT period_start, period_end FROM ingestion_runs WHERE "
            "component = %s AND symbol = %s AND status = 'success'",
            (component, symbol),
        )
        return {(r[0], r[1]) for r in cur.fetchall()}
```

- [ ] **Step 11: Run the repository tests and watch them pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration -v -m db
```
Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add data/storage tests
git commit -m "feat: TimescaleDB schema, migrations, and idempotent storage layer"
```

---

### Task 3: Binance REST client with retry, backoff, and rate-limit handling

**Files:**
- Create: `data/collectors/__init__.py`, `data/collectors/binance_rest.py`
- Test: `tests/unit/test_binance_rest.py`

**Interfaces:**
- Consumes: `data.config.get_settings`.
- Produces:
  - `RateLimitedError(Exception)` with attribute `retry_after: float`
  - `BinanceRest(base_url: str | None = None, client: httpx.Client | None = None)` with methods `get(path: str, params: dict) -> Any`, `exchange_info(symbols: list[str]) -> dict`, `klines(symbol: str, start_ms: int, limit: int = 1000) -> list[list]`, `first_candle_time(symbol: str) -> datetime`
  - `backoff_delays(attempts: int, base: float = 0.5, cap: float = 30.0) -> list[float]`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_binance_rest.py`:
```python
from datetime import timezone

import httpx
import pytest
import respx

from data.collectors.binance_rest import (
    BinanceRest,
    RateLimitedError,
    backoff_delays,
)

BASE = "https://api.binance.com"


def test_backoff_grows_and_is_capped():
    # Without jitter the growth is deterministic and must be monotonic.
    delays = backoff_delays(6, base=0.5, cap=8.0, jitter=False)
    assert delays[0] == pytest.approx(0.5)
    assert delays == sorted(delays)
    assert max(delays) <= 8.0


def test_backoff_jitter_never_exceeds_the_cap():
    # Jitter must be applied inside the cap, not on top of it: a delay
    # above the cap is what turns a retry storm into a ban.
    for _ in range(200):
        assert max(backoff_delays(8, base=0.5, cap=8.0)) <= 8.0


@respx.mock
def test_retries_then_succeeds_on_500():
    route = respx.get(f"{BASE}/api/v3/ping").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    assert api.get("/api/v3/ping", {}) == {"ok": True}
    assert route.call_count == 3


@respx.mock
def test_429_raises_rate_limited_with_retry_after():
    respx.get(f"{BASE}/api/v3/ping").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "120"})
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    with pytest.raises(RateLimitedError) as exc:
        api.get("/api/v3/ping", {})
    assert exc.value.retry_after == 120.0


@respx.mock
def test_418_ban_raises_immediately_without_retrying():
    route = respx.get(f"{BASE}/api/v3/ping").mock(
        return_value=httpx.Response(418, headers={"Retry-After": "300"})
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    with pytest.raises(RateLimitedError):
        api.get("/api/v3/ping", {})
    assert route.call_count == 1


@respx.mock
def test_first_candle_time_is_utc():
    respx.get(f"{BASE}/api/v3/klines").mock(
        return_value=httpx.Response(
            200,
            json=[[1502942400000, "4261.48", "4745.42", "4200.74", "4724.89",
                   "1000.5", 1502942459999, "4000000.0", 100, "500.2",
                   "2000000.0", "0"]],
        )
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    ts = api.first_candle_time("BTCUSDT")
    assert ts.tzinfo is timezone.utc
    assert ts.year == 2017
```

- [ ] **Step 2: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_binance_rest.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the client**

`data/collectors/__init__.py`: empty file.

`data/collectors/binance_rest.py`:
```python
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from data.config import get_settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {500, 502, 503, 504}


class RateLimitedError(Exception):
    def __init__(self, status: int, retry_after: float):
        super().__init__(f"rate limited (HTTP {status}); retry after {retry_after}s")
        self.status = status
        self.retry_after = retry_after


def backoff_delays(attempts: int, base: float = 0.5, cap: float = 30.0,
                   jitter: bool = True) -> list[float]:
    """Exponential backoff with jitter, capped.

    The jitter is applied *inside* the cap: a delay longer than the cap
    would be a surprise, and the cap is what keeps a retry storm bounded.
    """
    out = []
    for i in range(attempts):
        raw = base * (2 ** i)
        if jitter and i:
            raw *= 0.8 + 0.4 * random.random()
        out.append(min(cap, raw))
    return out


class BinanceRest:
    def __init__(self, base_url: str | None = None,
                 client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 attempts: int = 5):
        self.base_url = (base_url or get_settings().binance_base_url).rstrip("/")
        self.client = client or httpx.Client(timeout=30.0)
        self.sleep = sleep
        self.attempts = attempts

    def get(self, path: str, params: dict) -> Any:
        delays = backoff_delays(self.attempts)
        last_exc: Exception | None = None
        for attempt, delay in enumerate(delays):
            resp = self.client.get(self.base_url + path, params=params)
            if resp.status_code in (429, 418):
                # 418 means we are already banned: never retry into a ban.
                retry_after = float(resp.headers.get("Retry-After", 60))
                raise RateLimitedError(resp.status_code, retry_after)
            if resp.status_code in RETRYABLE_STATUS:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                log.warning("binance %s -> %s, retrying in %.1fs",
                            path, resp.status_code, delay)
                self.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise last_exc if last_exc else RuntimeError("request failed")

    def exchange_info(self, symbols: list[str]) -> dict:
        quoted = "[" + ",".join(f'"{s}"' for s in symbols) + "]"
        return self.get("/api/v3/exchangeInfo", {"symbols": quoted})

    def klines(self, symbol: str, start_ms: int, limit: int = 1000) -> list[list]:
        return self.get("/api/v3/klines", {
            "symbol": symbol, "interval": "1m",
            "startTime": start_ms, "limit": limit,
        })

    def first_candle_time(self, symbol: str) -> datetime:
        rows = self.klines(symbol, start_ms=0, limit=1)
        if not rows:
            raise ValueError(f"no klines returned for {symbol}")
        return datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc)
```

- [ ] **Step 4: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_binance_rest.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Add the symbols-sync function and its test**

Append to `tests/unit/test_binance_rest.py`:
```python
@respx.mock
def test_symbol_metadata_extracts_filters():
    from data.collectors.binance_rest import parse_symbol_info

    payload = {"symbols": [{
        "symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT",
        "status": "TRADING",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.00001000"},
            {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
        ],
    }]}
    info = parse_symbol_info(payload)["BTCUSDT"]
    assert info["tick_size"] == Decimal("0.01")
    assert info["step_size"] == Decimal("0.00001")
    assert info["min_notional"] == Decimal("5")
    assert info["status"] == "TRADING"
```
Add `from decimal import Decimal` to the imports at the top of the file.

- [ ] **Step 6: Run and watch it fail, then implement**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_binance_rest.py::test_symbol_metadata_extracts_filters -v
```
Expected: FAIL — `cannot import name 'parse_symbol_info'`.

Append to `data/collectors/binance_rest.py`:
```python
from decimal import Decimal


def parse_symbol_info(payload: dict) -> dict[str, dict]:
    """Flatten exchangeInfo into one dict per symbol."""
    out: dict[str, dict] = {}
    for s in payload.get("symbols", []):
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        out[s["symbol"]] = {
            "symbol": s["symbol"],
            "base_asset": s.get("baseAsset"),
            "quote_asset": s.get("quoteAsset"),
            "status": s.get("status"),
            "tick_size": Decimal(filters["PRICE_FILTER"]["tickSize"])
            if "PRICE_FILTER" in filters else None,
            "step_size": Decimal(filters["LOT_SIZE"]["stepSize"])
            if "LOT_SIZE" in filters else None,
            "min_notional": Decimal(filters["NOTIONAL"]["minNotional"])
            if "NOTIONAL" in filters else None,
        }
    return out
```

- [ ] **Step 7: Run the whole unit suite and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit -v
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add data/collectors tests/unit/test_binance_rest.py
git commit -m "feat: Binance REST client with backoff and rate-limit handling"
```

---

### Task 4: Archive parsing and checksum verification

**Files:**
- Create: `data/collectors/archive.py`
- Create: `tests/fixtures/BTCUSDT-1m-2024-01-sample.csv`, `tests/fixtures/BTCUSDT-1m-2025-01-sample.csv`
- Test: `tests/unit/test_archive.py`

**Interfaces:**
- Consumes: nothing (pure functions).
- Produces:
  - `monthly_url(base: str, symbol: str, year: int, month: int) -> str`
  - `daily_url(base: str, symbol: str, day: date) -> str`
  - `verify_sha256(data: bytes, checksum_text: str) -> bool`
  - `parse_timestamp(raw: str | int) -> datetime` (handles ms **and** µs)
  - `parse_kline_csv(text: str, symbol: str, source: str = "archive") -> list[Candle]`
  - `rows_from_zip(blob: bytes, symbol: str) -> list[Candle]`

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/BTCUSDT-1m-2024-01-sample.csv` (no header, milliseconds — the pre-2025 format):
```csv
1704067200000,42283.58000000,42554.57000000,42261.02000000,42475.23000000,1271.51075000,1704067259999,53948745.14724050,52891,657.85980000,27909858.09930610,0
1704067260000,42475.23000000,42500.00000000,42400.00000000,42450.00000000,300.00000001,1704067319999,12735000.00000000,1200,150.00000000,6367500.00000000,0
```

`tests/fixtures/BTCUSDT-1m-2025-01-sample.csv` (header row, microseconds — the 2025+ format):
```csv
open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
1735689600000000,93429.31000000,93500.00000000,93400.00000000,93480.00000000,55.12345678,1735689659999999,5152000.00000000,3021,30.00000000,2804400.00000000,0
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_archive.py`:
```python
import hashlib
import io
import zipfile
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from data.collectors.archive import (
    daily_url,
    monthly_url,
    parse_kline_csv,
    parse_timestamp,
    rows_from_zip,
    verify_sha256,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
BASE = "https://data.binance.vision"


def test_monthly_url():
    assert monthly_url(BASE, "BTCUSDT", 2024, 1) == (
        f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )


def test_daily_url():
    assert daily_url(BASE, "BTCUSDT", date(2024, 3, 5)) == (
        f"{BASE}/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03-05.zip"
    )


def test_verify_sha256_accepts_matching_and_rejects_tampered():
    blob = b"hello binance"
    digest = hashlib.sha256(blob).hexdigest()
    checksum_text = f"{digest}  BTCUSDT-1m-2024-01.zip\n"
    assert verify_sha256(blob, checksum_text) is True
    assert verify_sha256(b"tampered", checksum_text) is False


def test_parse_timestamp_handles_milliseconds_and_microseconds():
    # 2024-01-01T00:00:00Z in ms and in us must give the same instant.
    ms = parse_timestamp(1704067200000)
    us = parse_timestamp(1704067200000000)
    assert ms == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert us == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_parses_legacy_millisecond_file():
    text = (FIXTURES / "BTCUSDT-1m-2024-01-sample.csv").read_text()
    candles = parse_kline_csv(text, "BTCUSDT")
    assert len(candles) == 2
    first = candles[0]
    assert first.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert first.close == Decimal("42475.23000000")
    assert first.trade_count == 52891
    assert first.source == "archive"


def test_parses_2025_header_and_microsecond_file():
    text = (FIXTURES / "BTCUSDT-1m-2025-01-sample.csv").read_text()
    candles = parse_kline_csv(text, "BTCUSDT")
    assert len(candles) == 1
    assert candles[0].open_time == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert candles[0].close == Decimal("93480.00000000")


def test_preserves_decimal_precision_exactly():
    text = (FIXTURES / "BTCUSDT-1m-2024-01-sample.csv").read_text()
    candles = parse_kline_csv(text, "BTCUSDT")
    # 300.00000001 must survive; a float round-trip would not.
    assert candles[1].volume == Decimal("300.00000001")
    assert isinstance(candles[1].volume, Decimal)


def test_rows_from_zip_reads_the_single_member():
    text = (FIXTURES / "BTCUSDT-1m-2024-01-sample.csv").read_text()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1m-2024-01.csv", text)
    candles = rows_from_zip(buf.getvalue(), "BTCUSDT")
    assert len(candles) == 2


def test_rejects_row_with_wrong_column_count():
    with pytest.raises(ValueError, match="columns"):
        parse_kline_csv("1,2,3\n", "BTCUSDT")
```

- [ ] **Step 3: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_archive.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 4: Implement the archive module**

`data/collectors/archive.py`:
```python
import csv
import hashlib
import io
import zipfile
from datetime import date, datetime, timezone
from decimal import Decimal

from data.storage.repository import Candle

EXPECTED_COLUMNS = 12
# Binance switched archive timestamps from milliseconds to microseconds for
# files written from 2025 onward. 1e14 ms is year 5138, so any value above it
# is microseconds, not a plausible millisecond timestamp.
MICROSECOND_THRESHOLD = 1e14


def monthly_url(base: str, symbol: str, year: int, month: int) -> str:
    return (f"{base.rstrip('/')}/data/spot/monthly/klines/{symbol}/1m/"
            f"{symbol}-1m-{year:04d}-{month:02d}.zip")


def daily_url(base: str, symbol: str, day: date) -> str:
    return (f"{base.rstrip('/')}/data/spot/daily/klines/{symbol}/1m/"
            f"{symbol}-1m-{day:%Y-%m-%d}.zip")


def verify_sha256(data: bytes, checksum_text: str) -> bool:
    expected = checksum_text.strip().split()[0].lower()
    return hashlib.sha256(data).hexdigest() == expected


def parse_timestamp(raw: str | int) -> datetime:
    value = int(raw)
    seconds = value / 1_000_000 if value > MICROSECOND_THRESHOLD else value / 1_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def parse_kline_csv(text: str, symbol: str, source: str = "archive") -> list[Candle]:
    candles: list[Candle] = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0].strip().lower() == "open_time":
            continue  # header row, present in 2025+ archives
        if len(row) != EXPECTED_COLUMNS:
            raise ValueError(
                f"expected {EXPECTED_COLUMNS} columns, got {len(row)}: {row!r}"
            )
        candles.append(Candle(
            symbol=symbol,
            open_time=parse_timestamp(row[0]),
            close_time=parse_timestamp(row[6]),
            open=Decimal(row[1]),
            high=Decimal(row[2]),
            low=Decimal(row[3]),
            close=Decimal(row[4]),
            volume=Decimal(row[5]),
            quote_volume=Decimal(row[7]),
            trade_count=int(row[8]),
            taker_buy_base=Decimal(row[9]),
            taker_buy_quote=Decimal(row[10]),
            source=source,
        ))
    return candles


def rows_from_zip(blob: bytes, symbol: str, source: str = "archive") -> list[Candle]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in archive, found {names}")
        text = zf.read(names[0]).decode("utf-8")
    return parse_kline_csv(text, symbol, source=source)
```

- [ ] **Step 5: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_archive.py -v
```
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add data/collectors/archive.py tests/unit/test_archive.py tests/fixtures
git commit -m "feat: Binance archive parsing with checksum and us/ms timestamp handling"
```

---

### Task 5: Backfill orchestration

**Files:**
- Create: `data/collectors/backfill.py`
- Test: `tests/unit/test_backfill.py`, `tests/integration/test_backfill_db.py`

**Interfaces:**
- Consumes: `archive.*`, `BinanceRest`, `repository.*`.
- Produces:
  - `months_between(start: datetime, end: datetime) -> list[tuple[int, int]]`
  - `ArchiveDownloader(base_url, client=None, sleep=time.sleep)` with `fetch_month(symbol, year, month) -> bytes | None` (returns `None` on 404) and `fetch_day(symbol, day) -> bytes | None`
  - `backfill_symbol(conn, symbol: str, downloader, api: BinanceRest, until: datetime | None = None) -> int` (rows written)

- [ ] **Step 1: Write the failing unit tests**

`tests/unit/test_backfill.py`:
```python
from datetime import datetime, timezone

import httpx
import pytest
import respx

from data.collectors.backfill import ArchiveDownloader, months_between

BASE = "https://data.binance.vision"


def test_months_between_is_inclusive():
    start = datetime(2023, 11, 15, tzinfo=timezone.utc)
    end = datetime(2024, 2, 3, tzinfo=timezone.utc)
    assert months_between(start, end) == [
        (2023, 11), (2023, 12), (2024, 1), (2024, 2)
    ]


@respx.mock
def test_missing_month_returns_none_and_does_not_raise():
    """A month before the symbol was listed 404s. That is normal."""
    url = f"{BASE}/data/spot/monthly/klines/SOLUSDT/1m/SOLUSDT-1m-2017-01.zip"
    route = respx.get(url).mock(return_value=httpx.Response(404))
    dl = ArchiveDownloader(BASE, sleep=lambda _: None)

    assert dl.fetch_month("SOLUSDT", 2017, 1) is None
    assert route.call_count == 1  # not retried


@respx.mock
def test_checksum_mismatch_raises():
    url = f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    respx.get(url).mock(return_value=httpx.Response(200, content=b"payload"))
    respx.get(url + ".CHECKSUM").mock(
        return_value=httpx.Response(200, text="deadbeef  BTCUSDT-1m-2024-01.zip")
    )
    dl = ArchiveDownloader(BASE, sleep=lambda _: None)

    with pytest.raises(ValueError, match="checksum"):
        dl.fetch_month("BTCUSDT", 2024, 1)
```

- [ ] **Step 2: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_backfill.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the downloader and month helper**

`data/collectors/backfill.py`:
```python
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import httpx

from data.collectors import archive
from data.collectors.binance_rest import BinanceRest, backoff_delays
from data.storage.repository import (
    Candle,
    completed_periods,
    finish_run,
    last_candle_time,
    start_run,
    upsert_candles,
)

log = logging.getLogger(__name__)


def months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


class ArchiveDownloader:
    def __init__(self, base_url: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep, attempts: int = 4):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=120.0, follow_redirects=True)
        self.sleep = sleep
        self.attempts = attempts

    def _get(self, url: str) -> bytes | None:
        """Returns the body, or None for 404 (which means 'does not exist')."""
        for delay in backoff_delays(self.attempts):
            resp = self.client.get(url)
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                log.warning("%s -> %s, retrying in %.1fs", url, resp.status_code, delay)
                self.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.content
        raise RuntimeError(f"gave up downloading {url}")

    def _fetch_verified(self, url: str) -> bytes | None:
        blob = self._get(url)
        if blob is None:
            return None
        checksum = self._get(url + ".CHECKSUM")
        if checksum is None:
            raise ValueError(f"no checksum published for {url}")
        if not archive.verify_sha256(blob, checksum.decode("utf-8")):
            raise ValueError(f"checksum mismatch for {url}; refusing to load")
        return blob

    def fetch_month(self, symbol: str, year: int, month: int) -> bytes | None:
        return self._fetch_verified(
            archive.monthly_url(self.base_url, symbol, year, month)
        )

    def fetch_day(self, symbol: str, day: date) -> bytes | None:
        return self._fetch_verified(archive.daily_url(self.base_url, symbol, day))
```

- [ ] **Step 4: Run and watch the unit tests pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_backfill.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Write the failing orchestration test**

`tests/integration/test_backfill_db.py`:
```python
from datetime import datetime, timezone

import pytest

from data.collectors.backfill import backfill_symbol
from data.storage.repository import upsert_symbol

pytestmark = pytest.mark.db


class FakeDownloader:
    """Serves one month of data, 404s for everything else."""

    def __init__(self, blob_by_month):
        self.blob_by_month = blob_by_month
        self.calls = []

    def fetch_month(self, symbol, year, month):
        self.calls.append((symbol, year, month))
        return self.blob_by_month.get((year, month))

    def fetch_day(self, symbol, day):
        return None


class FakeApi:
    def first_candle_time(self, symbol):
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

    def klines(self, symbol, start_ms, limit=1000):
        return []


def make_zip(text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", text)
    return buf.getvalue()


CSV = (
    "1704067200000,42283.58,42554.57,42261.02,42475.23,1271.51075,"
    "1704067259999,53948745.14,52891,657.8598,27909858.09,0\n"
)


def test_backfill_loads_and_is_resumable(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = FakeDownloader({(2024, 1): make_zip(CSV)})

    # `until` must be past the end of January, or the month is treated as
    # incomplete and left to the daily/REST tail instead of the archive.
    written = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 2, 5, tzinfo=timezone.utc),
    )
    assert written == 1

    first_call_count = len(dl.calls)
    written_again = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 2, 5, tzinfo=timezone.utc),
    )
    assert written_again == 0, "already-loaded months must be skipped"
    assert len(dl.calls) == first_call_count, "no re-download on second run"


def test_backfill_records_runs(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = FakeDownloader({(2024, 1): make_zip(CSV)})
    backfill_symbol(db_conn, "BTCUSDT", dl, FakeApi(),
                    until=datetime(2024, 2, 5, tzinfo=timezone.utc))

    with db_conn.cursor() as cur:
        cur.execute("SELECT component, status FROM ingestion_runs")
        rows = cur.fetchall()
    assert ("backfill", "success") in rows
```

- [ ] **Step 6: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_backfill_db.py -v -m db
```
Expected: FAIL — `cannot import name 'backfill_symbol'`.

- [ ] **Step 7: Implement the orchestration**

Append to `data/collectors/backfill.py`:
```python
def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12
           else datetime(year, month + 1, 1, tzinfo=timezone.utc))
    return start, end


def backfill_symbol(conn, symbol: str, downloader, api,
                    until: datetime | None = None) -> int:
    """Load monthly archives, then daily archives, then the REST tail.

    Resumable: months already recorded as successful runs are skipped.
    Months that predate the symbol's listing 404 and are skipped quietly.
    """
    until = until or datetime.now(timezone.utc)
    start = api.first_candle_time(symbol)
    done = completed_periods(conn, "backfill", symbol)
    total = 0

    for year, month in months_between(start, until):
        m_start, m_end = _month_bounds(year, month)
        if (m_start, m_end) in done:
            continue
        if m_end > until:
            break  # incomplete month: handled by daily archives / REST below
        run_id = start_run(conn, "backfill", symbol, m_start, m_end)
        try:
            blob = downloader.fetch_month(symbol, year, month)
            if blob is None:
                # Before listing, or not published yet. Not an error.
                finish_run(conn, run_id, "success", 0)
                continue
            candles = archive.rows_from_zip(blob, symbol)
            written = upsert_candles(conn, candles)
            finish_run(conn, run_id, "success", written)
            total += written
            log.info("%s %04d-%02d: %d candles", symbol, year, month, written)
        except Exception as exc:
            finish_run(conn, run_id, "failed", 0, str(exc))
            log.error("%s %04d-%02d failed: %s", symbol, year, month, exc)

    total += _backfill_tail(conn, symbol, downloader, api, until)
    return total


def _backfill_tail(conn, symbol: str, downloader, api, until: datetime) -> int:
    """Fill from the last stored candle to `until` using daily archives, then
    the REST API for whatever is too recent to be archived."""
    written = 0
    last = last_candle_time(conn, symbol)
    cursor = (last + timedelta(minutes=1)) if last else None
    if cursor is None:
        return 0

    day = cursor.date()
    while day < until.date():
        blob = downloader.fetch_day(symbol, day)
        if blob is not None:
            candles = [c for c in archive.rows_from_zip(blob, symbol)
                       if c.open_time >= cursor]
            written += upsert_candles(conn, candles)
        day += timedelta(days=1)

    last = last_candle_time(conn, symbol)
    cursor = (last + timedelta(minutes=1)) if last else cursor
    while cursor < until:
        rows = api.klines(symbol, int(cursor.timestamp() * 1000), limit=1000)
        if not rows:
            break
        candles = [
            Candle(
                symbol=symbol,
                open_time=archive.parse_timestamp(r[0]),
                close_time=archive.parse_timestamp(r[6]),
                open=Decimal(r[1]), high=Decimal(r[2]), low=Decimal(r[3]),
                close=Decimal(r[4]), volume=Decimal(r[5]),
                quote_volume=Decimal(r[7]), trade_count=int(r[8]),
                taker_buy_base=Decimal(r[9]), taker_buy_quote=Decimal(r[10]),
                source="rest",
            )
            for r in rows
        ]
        # Drop the still-open final candle: only closed candles are stored.
        candles = [c for c in candles if c.close_time < until]
        if not candles:
            break
        written += upsert_candles(conn, candles)
        cursor = candles[-1].open_time + timedelta(minutes=1)
    return written
```
Add `from decimal import Decimal` to the module's imports.

- [ ] **Step 8: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_backfill_db.py -v -m db
```
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add data/collectors/backfill.py tests
git commit -m "feat: resumable historical backfill from Binance archives and REST"
```

---

### Task 6: Live collector

**Files:**
- Create: `data/collectors/binance_ws.py`, `data/collectors/live.py`
- Test: `tests/unit/test_binance_ws.py`, `tests/unit/test_live.py`

**Interfaces:**
- Consumes: `repository.upsert_candles`, `repository.upsert_book_ticker`, `BinanceRest`.
- Produces:
  - `stream_url(ws_base: str, symbols: list[str]) -> str`
  - `parse_kline_message(msg: dict, symbol_hint: str | None = None) -> Candle | None` (returns `None` for unclosed candles)
  - `parse_book_ticker_message(msg: dict, received_at: datetime) -> tuple`
  - `minutes_missing(last_open: datetime, now: datetime) -> int`
  - `LiveCollector(settings=None, api=None, connect_fn=connect)` with
    `async run(stop_event: asyncio.Event | None = None) -> None` and the
    synchronous `gap_fill(conn, symbol) -> int` (synchronous because psycopg
    and the REST client are both blocking; it runs before the socket opens)

- [ ] **Step 1: Write the failing parser tests**

`tests/unit/test_binance_ws.py`:
```python
from datetime import datetime, timezone
from decimal import Decimal

from data.collectors.binance_ws import (
    parse_book_ticker_message,
    parse_kline_message,
    stream_url,
)

CLOSED = {
    "stream": "btcusdt@kline_1m",
    "data": {"e": "kline", "E": 1704067259999, "s": "BTCUSDT", "k": {
        "t": 1704067200000, "T": 1704067259999, "s": "BTCUSDT", "i": "1m",
        "o": "42283.58", "c": "42475.23", "h": "42554.57", "l": "42261.02",
        "v": "1271.51075", "n": 52891, "x": True, "q": "53948745.14",
        "V": "657.8598", "Q": "27909858.09"}},
}


def test_stream_url_lowercases_and_joins():
    url = stream_url("wss://stream.binance.com:9443", ["BTCUSDT", "ETHUSDT"])
    assert url == (
        "wss://stream.binance.com:9443/stream?streams="
        "btcusdt@kline_1m/btcusdt@bookTicker/"
        "ethusdt@kline_1m/ethusdt@bookTicker"
    )


def test_parses_closed_candle():
    candle = parse_kline_message(CLOSED["data"])
    assert candle is not None
    assert candle.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert candle.close == Decimal("42475.23")
    assert candle.trade_count == 52891
    assert candle.source == "ws"


def test_ignores_unclosed_candle():
    msg = {"e": "kline", "s": "BTCUSDT", "k": dict(CLOSED["data"]["k"], x=False)}
    assert parse_kline_message(msg) is None


def test_parses_book_ticker():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    row = parse_book_ticker_message(
        {"u": 1, "s": "BTCUSDT", "b": "42000.01", "B": "1.5",
         "a": "42000.99", "A": "2.5"}, ts
    )
    assert row == ("BTCUSDT", ts, Decimal("42000.01"), Decimal("1.5"),
                   Decimal("42000.99"), Decimal("2.5"))
```

- [ ] **Step 2: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_binance_ws.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the WebSocket parsing module**

`data/collectors/binance_ws.py`:
```python
from datetime import datetime
from decimal import Decimal

from data.collectors.archive import parse_timestamp
from data.storage.repository import Candle


def stream_url(ws_base: str, symbols: list[str]) -> str:
    parts = []
    for s in symbols:
        low = s.lower()
        parts.append(f"{low}@kline_1m")
        parts.append(f"{low}@bookTicker")
    return f"{ws_base.rstrip('/')}/stream?streams=" + "/".join(parts)


def parse_kline_message(msg: dict, symbol_hint: str | None = None) -> Candle | None:
    """Returns a Candle only for CLOSED candles; None otherwise.

    A candle that is still forming will change before it closes, so storing it
    would put provisional data in the warehouse.
    """
    k = msg.get("k")
    if not k or not k.get("x"):
        return None
    return Candle(
        symbol=k.get("s") or msg.get("s") or symbol_hint,
        open_time=parse_timestamp(k["t"]),
        close_time=parse_timestamp(k["T"]),
        open=Decimal(k["o"]), high=Decimal(k["h"]), low=Decimal(k["l"]),
        close=Decimal(k["c"]), volume=Decimal(k["v"]),
        quote_volume=Decimal(k["q"]), trade_count=int(k["n"]),
        taker_buy_base=Decimal(k["V"]), taker_buy_quote=Decimal(k["Q"]),
        source="ws",
    )


def parse_book_ticker_message(msg: dict, received_at: datetime) -> tuple:
    return (
        msg["s"], received_at,
        Decimal(msg["b"]), Decimal(msg["B"]),
        Decimal(msg["a"]), Decimal(msg["A"]),
    )
```

- [ ] **Step 4: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_binance_ws.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Write the failing gap-backfill test**

`tests/unit/test_live.py`:
```python
from datetime import datetime, timedelta, timezone

import pytest

from data.collectors.live import minutes_missing


def test_minutes_missing_computes_the_gap():
    last = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    now = datetime(2024, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
    # candles at 12:01..12:04 are missing; 12:05 has not closed yet
    assert minutes_missing(last, now) == 4


def test_no_gap_when_current():
    last = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    now = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    assert minutes_missing(last, now) == 0


def test_requires_aware_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        minutes_missing(datetime(2024, 1, 1, 12, 0),
                        datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc))
```

- [ ] **Step 6: Run and watch it fail, then implement the live collector**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_live.py -v
```
Expected: FAIL — module not found.

`data/collectors/live.py`:
```python
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import websockets

from data.collectors.binance_rest import BinanceRest
from data.collectors.binance_ws import (
    parse_book_ticker_message,
    parse_kline_message,
    stream_url,
)
from data.config import get_settings
from data.storage.db import connect
from data.storage.repository import (
    Candle,
    last_candle_time,
    upsert_book_ticker,
    upsert_candles,
)
from data.collectors.archive import parse_timestamp
from decimal import Decimal

log = logging.getLogger(__name__)

BOOK_FLUSH_SECONDS = 5
MAX_BUFFERED_CANDLES = 10_000


def minutes_missing(last_open: datetime, now: datetime) -> int:
    """How many closed 1m candles are missing between last_open and now."""
    for label, dt in (("last_open", last_open), ("now", now)):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError(f"{label} must be timezone-aware UTC")
    last_closed = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
    return max(0, int((last_closed - last_open).total_seconds() // 60))


class LiveCollector:
    def __init__(self, settings=None, api: BinanceRest | None = None,
                 connect_fn=connect):
        self.settings = settings or get_settings()
        self.api = api or BinanceRest()
        self.connect_fn = connect_fn
        self.book_buffer: list[tuple] = []
        self.candle_buffer: list[Candle] = []
        self.last_message_at: datetime | None = None

    def gap_fill(self, conn, symbol: str) -> int:
        """Pull missed minutes from REST after a disconnect."""
        last = last_candle_time(conn, symbol)
        if last is None:
            return 0
        now = datetime.now(timezone.utc)
        if minutes_missing(last, now) == 0:
            return 0
        log.warning("gap detected for %s since %s; backfilling via REST",
                    symbol, last)
        written = 0
        cursor = last + timedelta(minutes=1)
        while cursor < now:
            rows = self.api.klines(symbol, int(cursor.timestamp() * 1000))
            if not rows:
                break
            candles = [
                Candle(
                    symbol=symbol,
                    open_time=parse_timestamp(r[0]),
                    close_time=parse_timestamp(r[6]),
                    open=Decimal(r[1]), high=Decimal(r[2]), low=Decimal(r[3]),
                    close=Decimal(r[4]), volume=Decimal(r[5]),
                    quote_volume=Decimal(r[7]), trade_count=int(r[8]),
                    taker_buy_base=Decimal(r[9]), taker_buy_quote=Decimal(r[10]),
                    source="rest",
                )
                for r in rows
            ]
            candles = [c for c in candles if c.close_time < now]
            if not candles:
                break
            written += upsert_candles(conn, candles)
            cursor = candles[-1].open_time + timedelta(minutes=1)
        return written

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        url = stream_url(self.settings.binance_ws_url, self.settings.symbols)
        while not stop_event.is_set():
            try:
                conn = self.connect_fn()
                for symbol in self.settings.symbols:
                    self.gap_fill(conn, symbol)
                async with websockets.connect(url, ping_interval=20) as ws:
                    log.info("live collector connected: %s", self.settings.symbols)
                    flusher = asyncio.create_task(self._flush_loop(conn, stop_event))
                    try:
                        async for raw in ws:
                            if stop_event.is_set():
                                break
                            self._handle(json.loads(raw), conn)
                    finally:
                        flusher.cancel()
                        conn.close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("live collector error: %s; reconnecting in 5s", exc)
                await asyncio.sleep(5)

    def _handle(self, message: dict, conn) -> None:
        data = message.get("data", message)
        self.last_message_at = datetime.now(timezone.utc)
        if data.get("e") == "kline":
            candle = parse_kline_message(data)
            if candle is not None:
                upsert_candles(conn, [candle])
        elif "b" in data and "a" in data:
            self.book_buffer.append(
                parse_book_ticker_message(data, self.last_message_at)
            )
            if len(self.book_buffer) > MAX_BUFFERED_CANDLES:
                log.error("book ticker buffer overflow; dropping %d rows",
                          len(self.book_buffer))
                self.book_buffer.clear()

    async def _flush_loop(self, conn, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(BOOK_FLUSH_SECONDS)
            if self.book_buffer:
                rows, self.book_buffer = self.book_buffer, []
                try:
                    upsert_book_ticker(conn, rows)
                except Exception as exc:
                    log.error("book ticker flush failed: %s", exc)
```

- [ ] **Step 7: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_live.py -v
```
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add data/collectors/binance_ws.py data/collectors/live.py tests/unit
git commit -m "feat: live WebSocket collector with reconnect and REST gap backfill"
```

---

### Task 7: Aggregation correctness proof

**Files:**
- Test: `tests/integration/test_aggregates.py`

**Interfaces:**
- Consumes: `repository.upsert_candles`, `repository.get_candles`.
- Produces: nothing; this task exists to prove the database's continuous aggregates match an independent computation. Sub-project 2 builds features on the aggregates, so an error here silently corrupts every model later.

- [ ] **Step 1: Write the failing test**

`tests/integration/test_aggregates.py`:
```python
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.storage.repository import Candle, get_candles, upsert_candles, upsert_symbol

pytestmark = pytest.mark.db

T0 = datetime(2024, 3, 1, tzinfo=timezone.utc)


def _random_candles(n: int) -> list[Candle]:
    random.seed(7)
    out = []
    for i in range(n):
        base = Decimal(random.randint(40000_00, 45000_00)) / 100
        high = base + Decimal("15.5")
        low = base - Decimal("12.25")
        open_time = T0 + timedelta(minutes=i)
        out.append(Candle(
            symbol="BTCUSDT", open_time=open_time,
            close_time=open_time + timedelta(seconds=59),
            open=base, high=high, low=low, close=base + Decimal("1.5"),
            volume=Decimal(i + 1), quote_volume=Decimal((i + 1) * 100),
            trade_count=i + 1, taker_buy_base=Decimal(i),
            taker_buy_quote=Decimal(i * 10), source="archive",
        ))
    return out


def _expected_hourly(candles: list[Candle]) -> dict:
    """Independent hourly aggregation in plain Python, exact on Decimals."""
    buckets: dict[datetime, list[Candle]] = defaultdict(list)
    for c in candles:
        buckets[c.open_time.replace(minute=0, second=0, microsecond=0)].append(c)
    out = {}
    for hour, group in buckets.items():
        group.sort(key=lambda c: c.open_time)
        out[hour] = {
            "open": group[0].open,
            "high": max(c.high for c in group),
            "low": min(c.low for c in group),
            "close": group[-1].close,
            "volume": sum((c.volume for c in group), Decimal(0)),
            "trade_count": sum(c.trade_count for c in group),
        }
    return out


def test_database_1h_candles_match_independent_computation(db_conn, autocommit_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    candles = _random_candles(180)  # exactly 3 hours
    upsert_candles(db_conn, candles)

    # Must run outside a transaction block, hence the autocommit connection.
    with autocommit_conn.cursor() as cur:
        cur.execute("CALL refresh_continuous_aggregate('candles_1h', NULL, NULL)")

    rows = get_candles(db_conn, "BTCUSDT", "1h", T0, T0 + timedelta(hours=3))
    assert len(rows) == 3

    expected = _expected_hourly(candles)
    for row in rows:
        want = expected[row["open_time"]]
        assert row["open"] == want["open"]
        assert row["high"] == want["high"]
        assert row["low"] == want["low"]
        assert row["close"] == want["close"]
        assert row["volume"] == want["volume"]
        assert row["trade_count"] == want["trade_count"]
```

- [ ] **Step 2: Run it**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_aggregates.py -v -m db
```
Expected: PASS if Task 2's aggregate definitions are right. If it fails, the bug is in the `first()`/`last()` ordering arguments in `001_initial.sql` — fix the migration, drop and recreate the test database, and rerun. **Do not adjust the test to match the database.**

To recreate the test database after changing a migration:
```bash
docker exec tb-timescaledb psql -U tb -d trading_buddy -c "DROP DATABASE trading_buddy_test; CREATE DATABASE trading_buddy_test OWNER tb;"
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_aggregates.py
git commit -m "test: prove TimescaleDB 1h aggregates match an independent computation"
```

---

### Task 8: Data-quality engine

**Files:**
- Create: `data/quality/__init__.py`, `data/quality/checks.py`, `data/quality/report.py`
- Test: `tests/unit/test_quality_rules.py`, `tests/integration/test_quality_db.py`

**Interfaces:**
- Consumes: a database connection only. It queries `candles_1m` and the
  aggregate views directly rather than through `get_candles`, because it
  needs aggregate counts the row-returning helper does not provide.
- Produces:
  - `Report` dataclass: `symbol, timeframe, checked_from, checked_to, total_candles, duplicates, invalid, missing, completeness_pct, verdict, details: dict` plus `render() -> str`
  - `find_invalid(rows: list[dict]) -> list[dict]`
  - `find_gaps(times: list[datetime], step: timedelta) -> list[tuple[datetime, datetime]]`
  - `run_quality_checks(conn, symbol, timeframe="1m", start=None, end=None, known_outages=None) -> Report`
  - `assert_trainable(conn, symbols: list[str], timeframe="1m") -> None` (raises `DataQualityError`)

- [ ] **Step 1: Write the failing rule tests**

`tests/unit/test_quality_rules.py`:
```python
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from data.quality.checks import find_gaps, find_invalid, verdict_for

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
MIN = timedelta(minutes=1)


def row(**over):
    base = dict(open_time=T0, open=Decimal("10"), high=Decimal("12"),
                low=Decimal("9"), close=Decimal("11"), volume=Decimal("1"),
                trade_count=5)
    base.update(over)
    return base


def test_find_invalid_catches_high_below_low():
    bad = row(high=Decimal("8"), low=Decimal("9"))
    assert find_invalid([row(), bad]) == [bad]


def test_find_invalid_catches_high_below_close():
    bad = row(high=Decimal("10"), close=Decimal("11"))
    assert find_invalid([bad]) == [bad]


def test_find_invalid_catches_negative_volume():
    bad = row(volume=Decimal("-1"))
    assert find_invalid([bad]) == [bad]


def test_find_invalid_accepts_a_flat_candle():
    flat = row(open=Decimal("10"), high=Decimal("10"),
               low=Decimal("10"), close=Decimal("10"), volume=Decimal("0"))
    assert find_invalid([flat]) == []


def test_find_gaps_returns_missing_ranges():
    times = [T0, T0 + MIN, T0 + 5 * MIN]
    assert find_gaps(times, MIN) == [(T0 + 2 * MIN, T0 + 5 * MIN)]


def test_find_gaps_empty_when_contiguous():
    times = [T0 + i * MIN for i in range(10)]
    assert find_gaps(times, MIN) == []


def test_verdict_thresholds():
    assert verdict_for(completeness=100.0, invalid=0, duplicates=0) == "PASS"
    assert verdict_for(completeness=99.99, invalid=0, duplicates=0) == "PASS"
    assert verdict_for(completeness=99.0, invalid=0, duplicates=0) == "WARN"
    assert verdict_for(completeness=100.0, invalid=1, duplicates=0) == "FAIL"
    assert verdict_for(completeness=80.0, invalid=0, duplicates=0) == "FAIL"


def test_known_outage_minutes_do_not_count_as_missing():
    """Binance has had real multi-hour outages. Those minutes exist nowhere,
    so they must not keep assert_trainable red forever."""
    from data.quality.checks import subtract_known_outages

    gaps = [(T0 + 2 * MIN, T0 + 5 * MIN)]
    outages = [(T0 + 2 * MIN, T0 + 5 * MIN)]
    assert subtract_known_outages(gaps, outages) == []
```

- [ ] **Step 2: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_quality_rules.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the rules**

`data/quality/__init__.py`: empty file.

`data/quality/checks.py`:
```python
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from data.quality.report import Report

STEP = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15), "1h": timedelta(hours=1),
        "4h": timedelta(hours=4), "1d": timedelta(days=1)}

PASS_COMPLETENESS = 99.9
WARN_COMPLETENESS = 95.0


class DataQualityError(Exception):
    pass


def find_invalid(rows: list[dict]) -> list[dict]:
    """Rows that violate the arithmetic every candle must satisfy."""
    bad = []
    for r in rows:
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        if h < l or h < o or h < c or l > o or l > c:
            bad.append(r)
        elif r["volume"] is None or r["volume"] < 0:
            bad.append(r)
        elif r.get("trade_count") is not None and r["trade_count"] < 0:
            bad.append(r)
        elif any(v is None for v in (o, h, l, c)):
            bad.append(r)
    return bad


def find_gaps(times: list[datetime], step: timedelta) -> list[tuple[datetime, datetime]]:
    """Ranges [gap_start, next_present) where candles are missing."""
    gaps = []
    for prev, nxt in zip(times, times[1:]):
        if nxt - prev > step:
            gaps.append((prev + step, nxt))
    return gaps


def subtract_known_outages(gaps, outages) -> list[tuple[datetime, datetime]]:
    """Drop gaps fully covered by a known exchange outage."""
    remaining = []
    for start, end in gaps:
        if any(o_start <= start and end <= o_end for o_start, o_end in outages):
            continue
        remaining.append((start, end))
    return remaining


def verdict_for(completeness: float, invalid: int, duplicates: int) -> str:
    if invalid or duplicates:
        return "FAIL"
    if completeness >= PASS_COMPLETENESS:
        return "PASS"
    if completeness >= WARN_COMPLETENESS:
        return "WARN"
    return "FAIL"


def run_quality_checks(conn, symbol: str, timeframe: str = "1m",
                       start: datetime | None = None,
                       end: datetime | None = None,
                       known_outages: list[tuple] | None = None) -> Report:
    step = STEP[timeframe]
    table = "candles_1m" if timeframe == "1m" else f"candles_{timeframe}"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT min(open_time), max(open_time), count(*) FROM {table} "
            f"WHERE symbol = %s", (symbol,)
        )
        first, last, total = cur.fetchone()
        if not total:
            report = Report(symbol=symbol, timeframe=timeframe,
                            checked_from=None, checked_to=None,
                            total_candles=0, duplicates=0, invalid=0,
                            missing=0, completeness_pct=0.0, verdict="FAIL",
                            details={"reason": "no data"})
            _store(conn, report)
            return report

        start = start or first
        end = end or last
        cur.execute(
            f"SELECT open_time, open, high, low, close, volume, trade_count "
            f"FROM {table} WHERE symbol = %s AND open_time BETWEEN %s AND %s "
            f"ORDER BY open_time", (symbol, start, end)
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        # The primary key makes duplicates impossible in candles_1m; this
        # check exists so a future schema change cannot silently allow them.
        cur.execute(
            f"SELECT count(*) FROM (SELECT symbol, open_time FROM {table} "
            f"WHERE symbol = %s GROUP BY 1, 2 HAVING count(*) > 1) d", (symbol,)
        )
        duplicates = cur.fetchone()[0]

    times = [r["open_time"] for r in rows]
    out_of_order = sum(1 for a, b in zip(times, times[1:]) if b <= a)
    gaps = find_gaps(times, step)
    if known_outages:
        gaps = subtract_known_outages(gaps, known_outages)
    missing = sum(int((g_end - g_start) / step) for g_start, g_end in gaps)
    invalid = len(find_invalid(rows))

    expected = int((end - start) / step) + 1
    completeness = 100.0 * (expected - missing) / expected if expected else 0.0
    verdict = verdict_for(completeness, invalid, duplicates + out_of_order)

    report = Report(
        symbol=symbol, timeframe=timeframe, checked_from=start, checked_to=end,
        total_candles=len(rows), duplicates=duplicates, invalid=invalid,
        missing=missing, completeness_pct=round(completeness, 6),
        verdict=verdict,
        details={
            "out_of_order": out_of_order,
            "gap_count": len(gaps),
            "largest_gaps": [
                [g[0].isoformat(), g[1].isoformat()]
                for g in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:10]
            ],
        },
    )
    _store(conn, report)
    return report


def _store(conn, report: Report) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO data_quality_reports (symbol, timeframe, checked_from, "
            "checked_to, total_candles, duplicates, invalid, missing, "
            "completeness_pct, verdict, details) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (report.symbol, report.timeframe, report.checked_from,
             report.checked_to, report.total_candles, report.duplicates,
             report.invalid, report.missing, report.completeness_pct,
             report.verdict, json.dumps(report.details)),
        )
    conn.commit()


def assert_trainable(conn, symbols: list[str], timeframe: str = "1m") -> None:
    """Every later sub-project calls this before training. A FAIL stops it."""
    failures = []
    for symbol in symbols:
        report = run_quality_checks(conn, symbol, timeframe)
        if report.verdict == "FAIL":
            failures.append(report)
    if failures:
        summary = "\n".join(r.render() for r in failures)
        raise DataQualityError(
            f"data quality FAILED for {len(failures)} symbol(s); "
            f"refusing to train:\n{summary}"
        )
```

`data/quality/report.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Report:
    symbol: str
    timeframe: str
    checked_from: datetime | None
    checked_to: datetime | None
    total_candles: int
    duplicates: int
    invalid: int
    missing: int
    completeness_pct: float
    verdict: str
    details: dict = field(default_factory=dict)

    def render(self) -> str:
        return (
            f"\n{self.symbol}  [{self.timeframe}]\n"
            f"\nCandles:\n{self.total_candles:,}\n"
            f"\nDuplicates:\n{self.duplicates}\n"
            f"\nInvalid:\n{self.invalid}\n"
            f"\nMissing:\n{self.missing}\n"
            f"\nCompleteness:\n{self.completeness_pct:.6f}%\n"
            f"\nSTATUS:\n{self.verdict}\n"
        )
```

- [ ] **Step 4: Run the rule tests and watch them pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_quality_rules.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Write the failing end-to-end quality test**

`tests/integration/test_quality_db.py`:
```python
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.quality.checks import DataQualityError, assert_trainable, run_quality_checks
from data.storage.repository import Candle, upsert_candles, upsert_symbol

pytestmark = pytest.mark.db

T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)


def candle(minute: int, **over):
    open_time = T0 + timedelta(minutes=minute)
    base = dict(
        symbol="BTCUSDT", open_time=open_time,
        close_time=open_time + timedelta(seconds=59),
        open=Decimal("100"), high=Decimal("110"), low=Decimal("90"),
        close=Decimal("105"), volume=Decimal("1"), quote_volume=Decimal("105"),
        trade_count=3, taker_buy_base=Decimal("0.5"),
        taker_buy_quote=Decimal("52.5"), source="archive",
    )
    base.update(over)
    return Candle(**base)


def test_clean_data_passes(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [candle(i) for i in range(120)])

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "PASS"
    assert report.missing == 0
    assert report.total_candles == 120
    assert "STATUS" in report.render()


def test_gap_reduces_completeness(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(200) if not (50 <= i < 100)]
    upsert_candles(db_conn, rows)

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.missing == 50
    assert report.completeness_pct < 100
    assert report.verdict in {"WARN", "FAIL"}


def test_invalid_candle_fails_and_blocks_training(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(10)]
    rows.append(candle(10, high=Decimal("50"), low=Decimal("90")))
    upsert_candles(db_conn, rows)

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "FAIL"
    assert report.invalid == 1

    with pytest.raises(DataQualityError, match="refusing to train"):
        assert_trainable(db_conn, ["BTCUSDT"])


def test_known_outage_is_not_counted_against_the_data(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(200) if not (50 <= i < 100)]
    upsert_candles(db_conn, rows)

    outage = [(T0 + timedelta(minutes=50), T0 + timedelta(minutes=100))]
    report = run_quality_checks(db_conn, "BTCUSDT", known_outages=outage)
    assert report.missing == 0
    assert report.verdict == "PASS"


def test_report_is_stored(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [candle(i) for i in range(10)])
    run_quality_checks(db_conn, "BTCUSDT")

    with db_conn.cursor() as cur:
        cur.execute("SELECT symbol, verdict FROM data_quality_reports")
        assert cur.fetchone()[0] == "BTCUSDT"
```

- [ ] **Step 6: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_quality_db.py -v -m db
```
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add data/quality tests
git commit -m "feat: data-quality engine with PASS/WARN/FAIL verdicts and training gate"
```

---

### Task 9: CLI wiring and the operator runbook

**Files:**
- Create: `data/cli.py`, `README.md`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: everything built so far.
- Produces: the `tb` command with sub-commands `db upgrade`, `symbols sync`, `backfill`, `live`, `quality`, `status`.

- [ ] **Step 1: Write the failing CLI test**

`tests/unit/test_cli.py`:
```python
from typer.testing import CliRunner

from data.cli import app

runner = CliRunner()


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["db", "symbols", "backfill", "live", "quality", "status"]:
        assert cmd in result.output


def test_quality_command_rejects_unknown_timeframe():
    # Must fail on the argument, not on missing configuration: the
    # validation runs before any settings are loaded.
    result = runner.invoke(app, ["quality", "--timeframe", "7m"])
    assert result.exit_code != 0
    assert "7m" in result.output or "timeframe" in result.output.lower()
```

- [ ] **Step 2: Run and watch it fail**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_cli.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the CLI**

`data/cli.py`:
```python
import asyncio
import logging
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from data.collectors.backfill import ArchiveDownloader, backfill_symbol
from data.collectors.binance_rest import BinanceRest, parse_symbol_info
from data.collectors.live import LiveCollector
from data.config import get_settings
from data.quality.checks import STEP, run_quality_checks
from data.storage.db import connect, run_migrations
from data.storage.repository import last_candle_time, upsert_symbol

app = typer.Typer(help="AI Trading Buddy data foundation")
db_app = typer.Typer(help="Database maintenance")
symbols_app = typer.Typer(help="Symbol metadata")
app.add_typer(db_app, name="db")
app.add_typer(symbols_app, name="symbols")
console = Console()

# A 1m feed more than this far behind is stale, not merely quiet.
STALE_AFTER_MINUTES = 5


def _setup_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@db_app.command("upgrade")
def db_upgrade():
    """Apply database migrations."""
    _setup_logging()
    applied = run_migrations()
    console.print(f"applied: {applied or 'nothing new'}")


@symbols_app.command("sync")
def symbols_sync():
    """Refresh symbol metadata and listing dates from Binance."""
    _setup_logging()
    settings = get_settings()
    api = BinanceRest()
    info = parse_symbol_info(api.exchange_info(settings.symbols))
    with connect() as conn:
        for symbol, fields in info.items():
            fields["listed_at"] = api.first_candle_time(symbol)
            upsert_symbol(conn, **fields)
            console.print(f"{symbol}: listed {fields['listed_at']:%Y-%m-%d}")


@app.command()
def backfill(
    symbol: str = typer.Option(None, help="One symbol; default is all configured"),
):
    """Load historical 1m candles. Resumable: safe to re-run."""
    _setup_logging()
    settings = get_settings()
    targets = [symbol.upper()] if symbol else settings.symbols
    api = BinanceRest()
    downloader = ArchiveDownloader(settings.binance_data_url)
    with connect() as conn:
        for s in targets:
            written = backfill_symbol(conn, s, downloader, api)
            console.print(f"{s}: {written:,} candles written")


@app.command()
def live():
    """Run the live collector until interrupted."""
    _setup_logging()
    collector = LiveCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        console.print("stopped")


@app.command()
def quality(
    symbol: str = typer.Option(None),
    timeframe: str = typer.Option("1m"),
):
    """Run data-quality checks and print the report."""
    # Validate arguments before touching configuration, so a bad argument
    # reports itself rather than a confusing config error.
    if timeframe not in STEP:
        raise typer.BadParameter(f"unknown timeframe {timeframe!r}")
    _setup_logging()
    settings = get_settings()
    targets = [symbol.upper()] if symbol else settings.symbols
    with connect() as conn:
        for s in targets:
            console.print(run_quality_checks(conn, s, timeframe).render())


@app.command()
def status():
    """Show row counts, last candle, and last quality verdict per symbol."""
    _setup_logging()
    settings = get_settings()
    now = datetime.now(timezone.utc)
    table = Table("symbol", "candles", "last candle (UTC)", "age", "last verdict")
    with connect() as conn:
        for s in settings.symbols:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM candles_1m WHERE symbol=%s", (s,))
                count = cur.fetchone()[0]
                cur.execute(
                    "SELECT verdict FROM data_quality_reports WHERE symbol=%s "
                    "ORDER BY created_at DESC LIMIT 1", (s,)
                )
                row = cur.fetchone()
            last = last_candle_time(conn, s)
            if last is None:
                age = "-"
            else:
                minutes = int((now - last).total_seconds() // 60)
                # More than STALE_AFTER_MINUTES behind means the live
                # collector is down or the feed is stuck.
                age = (f"[red]{minutes}m STALE[/red]"
                       if minutes > STALE_AFTER_MINUTES else f"{minutes}m")
            table.add_row(
                s, f"{count:,}",
                last.strftime("%Y-%m-%d %H:%M") if last else "-",
                age,
                row[0] if row else "-",
            )
    console.print(table)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run and watch it pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_cli.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Run the entire suite**

Run:
```bash
.venv\Scripts\python.exe -m pytest -v
```
Expected: every test passes; network tests deselected.

Then with the database tests included:
```bash
.venv\Scripts\python.exe -m pytest -v -m "db or not db"
```

- [ ] **Step 6: Write the README**

`README.md`:
````markdown
# AI Trading Buddy — Data Foundation

Verified Binance market-data warehouse. Sub-project 1 of the platform;
see `docs/superpowers/specs/` for the design and `docs/superpowers/plans/`
for the implementation plan.

## First-time setup

1. Install Docker Desktop and confirm `docker info` works.
2. Copy `.env.example` to `.env`.
3. Start the database:

   ```bash
   docker compose up -d
   ```

4. Create the virtual environment and install:

   ```bash
   py -3.13 -m venv .venv
   ```

   ```bash
   .venv\Scripts\python.exe -m pip install -e ".[dev]"
   ```

5. Apply migrations:

   ```bash
   .venv\Scripts\tb.exe db upgrade
   ```

## Daily use

```bash
.venv\Scripts\tb.exe symbols sync
```

```bash
.venv\Scripts\tb.exe backfill
```

```bash
.venv\Scripts\tb.exe live
```

```bash
.venv\Scripts\tb.exe quality
```

```bash
.venv\Scripts\tb.exe status
```

The first `backfill` downloads years of 1-minute candles for four symbols
and takes a while. It is resumable: stop it with Ctrl+C and re-run.

## Facts worth knowing

- All timestamps are UTC. All prices are `Decimal`, never `float`.
- Only **closed** candles are stored. A forming candle is never written.
- `candles_1m` is the only source of truth; 5m/15m/1h/4h/1d are derived by
  TimescaleDB and cannot disagree with it.
- Source precedence is `archive` > `rest` > `ws`; a lower source never
  overwrites a higher one.
- `assert_trainable()` in `data/quality/checks.py` is the gate every future
  training run must call first.

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

Network tests are skipped by default; run them deliberately with
`-m network`.
````

- [ ] **Step 7: Commit**

```bash
git add data/cli.py README.md tests/unit/test_cli.py
git commit -m "feat: tb CLI and operator runbook"
```

- [ ] **Step 8: Real-data smoke run**

This is the moment the sub-project proves itself against reality rather than fixtures.

Run:
```bash
.venv\Scripts\tb.exe symbols sync
```
Expected: four symbols with plausible listing dates (BTCUSDT 2017-08-17, ETHUSDT 2017-08-17, BNBUSDT 2017-11-06, SOLUSDT 2020-08-11).

Then backfill one symbol first, to catch problems cheaply:
```bash
.venv\Scripts\tb.exe backfill --symbol SOLUSDT
```

Then:
```bash
.venv\Scripts\tb.exe quality --symbol SOLUSDT
```
Expected: a report with completeness above 99.9% and verdict PASS or WARN. Investigate any FAIL before backfilling the rest — a real problem found on one symbol is cheaper than on four.

Then the remaining symbols:
```bash
.venv\Scripts\tb.exe backfill
```

Finally, confirm the live collector runs and writes:
```bash
.venv\Scripts\tb.exe live
```
Let it run for three minutes, stop it with Ctrl+C, and check that `tb status` shows a last candle time within the last two minutes.

- [ ] **Step 9: Commit the findings**

```bash
git add -A
git commit -m "chore: record real-data smoke run results"
```

---

## Done when

- `pytest` is green, and `pytest -m db` is green.
- `tb backfill` has loaded full history for all four symbols.
- `tb quality` reports PASS or WARN (never FAIL) for all four.
- `tb live` runs, survives a deliberate network drop, and refills the gap.
- `tb status` shows current data.

Sub-project 2 (market intelligence: indicators, trend, volatility, regime)
starts from `get_candles()` and `assert_trainable()` and needs nothing else
from this code.
