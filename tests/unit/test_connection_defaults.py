"""No database connection may hang forever, and the default DSNs must not
depend on how `localhost` resolves.

The container is published on IPv4 loopback only (127.0.0.1:5433). On
Windows `localhost` resolves to ::1 first; nothing answers there and nothing
refuses, so a connection with no timeout waits forever.
"""
import ipaddress
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import psycopg

from data.storage import db, repository

ROOT = Path(__file__).parents[2]


class _Stop(Exception):
    """Raised by the fake connect so no real connection is needed."""


def _capture_connect(monkeypatch, module):
    calls = []

    def fake_connect(*args, **kwargs):
        calls.append((args, kwargs))
        raise _Stop

    monkeypatch.setattr(module.psycopg, "connect", fake_connect)
    return calls


def _timeout_of(args, kwargs):
    """The connect timeout, whether passed as a kwarg or inside the DSN."""
    if "connect_timeout" in kwargs:
        return int(kwargs["connect_timeout"])
    dsn = args[0] if args else ""
    found = re.search(r"connect_timeout=(\d+)", dsn)
    return int(found.group(1)) if found else None


def test_timeout_constant_is_ten_seconds():
    assert db.CONNECT_TIMEOUT_SECONDS == 10


def test_connect_sets_a_connect_timeout(monkeypatch):
    calls = _capture_connect(monkeypatch, db)
    try:
        db.connect("postgresql://u:p@127.0.0.1:5433/x")
    except _Stop:
        pass
    assert len(calls) == 1
    assert _timeout_of(*calls[0]) == db.CONNECT_TIMEOUT_SECONDS


def test_run_migrations_sets_a_connect_timeout(monkeypatch):
    calls = _capture_connect(monkeypatch, db)
    try:
        db.run_migrations("postgresql://u:p@127.0.0.1:5433/x")
    except _Stop:
        pass
    assert len(calls) == 1
    assert _timeout_of(*calls[0]) == db.CONNECT_TIMEOUT_SECONDS


def test_refresh_aggregates_connection_sets_a_connect_timeout(monkeypatch):
    calls = _capture_connect(monkeypatch, repository)
    conn = MagicMock()
    # Worst case: the parameters rebuilt from the caller's connection carry
    # no connect_timeout (e.g. a connection made without one).
    conn.info.get_parameters.return_value = {"dbname": "x", "host": "127.0.0.1",
                                             "port": "5433", "user": "u"}
    conn.info.password = "p"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    try:
        repository.refresh_aggregates(conn, start, start.replace(day=2),
                                      now=start.replace(year=2025))
    except _Stop:
        pass
    assert len(calls) == 1
    assert _timeout_of(*calls[0]) == db.CONNECT_TIMEOUT_SECONDS


def test_refresh_aggregates_keeps_a_timeout_the_connection_already_has(monkeypatch):
    calls = _capture_connect(monkeypatch, repository)
    conn = MagicMock()
    conn.info.get_parameters.return_value = {"dbname": "x", "connect_timeout": "3"}
    conn.info.password = "p"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    try:
        repository.refresh_aggregates(conn, start, start.replace(day=2),
                                      now=start.replace(year=2025))
    except _Stop:
        pass
    assert _timeout_of(*calls[0]) == 3


def test_no_other_connect_call_in_data_bypasses_the_timeout():
    """Every psycopg connect in data/ is one of the three audited sites."""
    sites = []
    for path in (ROOT / "data").rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"\b(psycopg\.connect|AsyncConnection\.connect)\(", line):
                sites.append(path.relative_to(ROOT).as_posix())
    assert sorted(sites) == ["data/storage/db.py", "data/storage/db.py",
                             "data/storage/repository.py"]


def _host(url: str) -> str:
    return urlsplit(url).hostname


def _is_ipv4_literal(host: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(host), ipaddress.IPv4Address)
    except ValueError:
        return False


def test_conftest_default_dsn_uses_an_ipv4_literal(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    import importlib
    import tests.conftest as conftest
    conftest = importlib.reload(conftest)
    assert _is_ipv4_literal(_host(conftest.TEST_DSN)), conftest.TEST_DSN.split("@")[-1]


def test_env_example_urls_use_an_ipv4_literal():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    urls = dict(re.findall(r"^(DATABASE_URL|TEST_DATABASE_URL)=(\S+)$", text,
                           re.MULTILINE))
    assert set(urls) == {"DATABASE_URL", "TEST_DATABASE_URL"}
    for name, url in urls.items():
        assert _is_ipv4_literal(_host(url)), f"{name} must not use a hostname"
