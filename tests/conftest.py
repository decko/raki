"""Shared test fixtures for RAKI tests.

Factory fixtures live here and are never duplicated across test files.
"""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sessions_dir(fixtures_dir: Path) -> Path:
    """Return the path to the session fixtures directory."""
    return fixtures_dir / "sessions"


@pytest.fixture
def pass_simple_dir(sessions_dir: Path) -> Path:
    """Return the path to the pass-simple session fixture."""
    return sessions_dir / "pass-simple"


@pytest.fixture
def rework_cycle_dir(sessions_dir: Path) -> Path:
    """Return the path to the rework-cycle session fixture."""
    return sessions_dir / "rework-cycle"


@pytest.fixture
def malformed_dir(sessions_dir: Path) -> Path:
    """Return the path to the malformed session fixture."""
    return sessions_dir / "malformed"
