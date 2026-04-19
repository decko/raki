"""Tests that verify example manifest files load correctly."""

from pathlib import Path

import pytest

from raki.ground_truth.manifest import EvalManifest, load_manifest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

MANIFEST_FILES = [
    "raki-minimal.yaml",
    "raki-full.yaml",
    "raki-alcove.yaml",
]


@pytest.fixture(scope="module", params=MANIFEST_FILES)
def loaded_manifest(request: pytest.FixtureRequest) -> tuple[str, EvalManifest]:
    """Load each example manifest once per module, returning (filename, manifest)."""
    manifest_name: str = request.param
    manifest_path = EXAMPLES_DIR / manifest_name
    manifest = load_manifest(manifest_path, project_root=EXAMPLES_DIR)
    return manifest_name, manifest


def test_example_manifest_parses(loaded_manifest: tuple[str, EvalManifest]) -> None:
    """Each example manifest must parse without errors."""
    manifest_name, manifest = loaded_manifest
    assert isinstance(manifest, EvalManifest), f"{manifest_name} did not produce an EvalManifest"


def test_example_manifest_has_sessions_path(loaded_manifest: tuple[str, EvalManifest]) -> None:
    """Each example manifest must have a resolved sessions path."""
    manifest_name, manifest = loaded_manifest
    assert manifest.sessions.path is not None, f"{manifest_name} missing sessions.path"
    assert manifest.sessions.path.exists(), f"{manifest_name} sessions.path does not exist"


def test_full_manifest_has_ground_truth_path() -> None:
    """raki-full.yaml must have ground_truth.path set."""
    manifest = load_manifest(EXAMPLES_DIR / "raki-full.yaml", project_root=EXAMPLES_DIR)
    assert manifest.ground_truth.path is not None, "raki-full.yaml missing ground_truth.path"
    assert manifest.ground_truth.path.exists(), "raki-full.yaml ground_truth.path does not exist"


def test_alcove_manifest_has_alcove_format() -> None:
    """raki-alcove.yaml must have format set to 'alcove'."""
    manifest = load_manifest(EXAMPLES_DIR / "raki-alcove.yaml", project_root=EXAMPLES_DIR)
    assert manifest.sessions.format == "alcove", (
        f"Expected format 'alcove', got '{manifest.sessions.format}'"
    )
