"""Shared test fixtures for RAKI tests.

Factory fixtures live here and are never duplicated across test files.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest

from raki.model import (
    EvalDataset,
    EvalSample,
    PhaseResult,
    ReviewFinding,
    SessionMeta,
)


def make_sample(
    session_id: str,
    rework_cycles: int = 0,
    cost: float = 10.0,
    verify_gen: int = 1,
    verify_status: Literal["completed", "failed", "skipped"] = "completed",
    findings: list[ReviewFinding] | None = None,
) -> EvalSample:
    meta = SessionMeta(
        session_id=session_id,
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=3,
        rework_cycles=rework_cycles,
        total_cost_usd=cost,
    )
    verify_output = "PASS" if verify_status == "completed" else "FAIL"
    phases = [
        PhaseResult(name="implement", generation=1, status="completed", output="done"),
        PhaseResult(
            name="verify",
            generation=verify_gen,
            status=verify_status,
            output=verify_output,
            output_structured={"verdict": verify_output},
        ),
    ]
    return EvalSample(
        session=meta,
        phases=phases,
        findings=findings or [],
        events=[],
    )


def make_dataset(*samples: EvalSample) -> EvalDataset:
    return EvalDataset(samples=list(samples))


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


@pytest.fixture
def manifest_with_session(tmp_path: Path, pass_simple_dir: Path) -> tuple[Path, Path]:
    """Create a tmp_path with a manifest and a copied pass-simple session.

    Returns:
        Tuple of (manifest_path, sessions_directory).
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session_dest = sessions / "101"
    session_dest.mkdir()
    for file_path in pass_simple_dir.iterdir():
        (session_dest / file_path.name).write_text(file_path.read_text())
    manifest = tmp_path / "raki.yaml"
    manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
    return manifest, sessions


@pytest.fixture
def empty_manifest(tmp_path: Path) -> Path:
    """Create a tmp_path with a manifest pointing to an empty sessions directory.

    Returns:
        Path to the manifest file.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    manifest = tmp_path / "raki.yaml"
    manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
    return manifest
