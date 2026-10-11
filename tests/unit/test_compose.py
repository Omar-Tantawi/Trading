"""docker-compose.yml is deployment configuration that moves to a VPS
unchanged, where Docker-published ports bypass ufw."""
import re
from pathlib import Path

COMPOSE = (Path(__file__).parents[2] / "docker-compose.yml").read_text(encoding="utf-8")


def test_database_port_is_published_on_loopback_only():
    ports = re.findall(r'^\s*-\s*"?([^"\s]+:5432)"?\s*$', COMPOSE, re.MULTILINE)
    assert ports == ["127.0.0.1:5433:5432"]


def test_password_comes_from_the_environment_with_the_existing_default():
    # An existing volume keeps the password it was initialised with, so the
    # default must stay the value this machine's volume was created with.
    assert re.search(r"POSTGRES_PASSWORD:\s*\$\{POSTGRES_PASSWORD:-tb_local_dev\}",
                     COMPOSE)
    assert not re.search(r"POSTGRES_PASSWORD:\s*tb_local_dev\s*$", COMPOSE,
                         re.MULTILINE), "no hard-coded password"


def test_every_top_level_package_is_installed():
    """tb.exe on the PC sees only the packages pyproject lists (an editable
    install); tests run from the repo root would not notice a missing one."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).parents[2]
    include = tomllib.loads((root / "pyproject.toml").read_text())[
        "tool"]["setuptools"]["packages"]["find"]["include"]
    packages = {p.parent.name for p in root.glob("*/__init__.py")} - {"tests"}
    for name in packages:
        assert f"{name}*" in include, f"{name} is missing from pyproject include"
