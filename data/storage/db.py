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
