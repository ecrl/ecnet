"""Shared pytest configuration for the ECNet test suite.

Marker registration lives in ``pyproject.toml`` under ``[tool.pytest.ini_options]``.
"""

from __future__ import annotations

import pytest

_PROPS = ["bp", "cn", "cp", "kv", "lhv", "mon", "pp", "ron", "ysi", "mp"]
_BACKEND = "padel"
_N_DESC = 1875
_N_PROCESSES = 1
_EPOCHS = 10


@pytest.fixture(scope="session")
def props() -> list[str]:
    return list(_PROPS)


@pytest.fixture(scope="session")
def backend() -> str:
    return _BACKEND


@pytest.fixture(scope="session")
def n_desc() -> int:
    return _N_DESC


@pytest.fixture(scope="session")
def n_processes() -> int:
    return _N_PROCESSES


@pytest.fixture(scope="session")
def epochs() -> int:
    return _EPOCHS
