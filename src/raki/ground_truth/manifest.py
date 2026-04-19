"""Manifest loader for RAKI evaluation configuration.

Loads and validates raki.yaml / eval-manifest.yaml files, resolving
relative paths and enforcing path traversal guards.
"""

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SessionFilter(BaseModel):
    """Filter criteria for selecting sessions to evaluate."""

    tickets: list[str] | None = None
    after: datetime | None = None
    min_phases: int = 0


class SessionsConfig(BaseModel):
    """Configuration for session data sources."""

    path: Path
    format: str = "auto"
    filter: SessionFilter = Field(default_factory=SessionFilter)


class SourceDocument(BaseModel):
    """Reference to a source document used for retrieval evaluation."""

    path: Path
    repo: str = ""
    domains: list[str] = Field(default_factory=list)


class GroundTruthConfig(BaseModel):
    """Configuration for ground truth data."""

    path: Path | None = None


class SyntheticConfig(BaseModel):
    """Configuration for synthetic test generation."""

    enabled: bool = False
    output: Path | None = None
    count: int = 50
    seed: int = 42


class EvalManifest(BaseModel):
    """Top-level evaluation manifest model.

    Represents the full contents of a raki.yaml or eval-manifest.yaml file,
    including session sources, ground truth, and synthetic generation config.
    """

    sessions: SessionsConfig
    sources: list[SourceDocument] = Field(default_factory=list)
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)


def _resolve_and_guard(
    field_path: Path,
    manifest_dir: Path,
    root: Path,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a manifest path and validate it stays within the project root.

    Args:
        field_path: The raw path from the manifest field.
        manifest_dir: Resolved parent directory of the manifest file.
        root: Resolved project root directory.
        label: Human-readable label for error messages (e.g. "sessions.path").
        must_exist: Whether the resolved path must exist on disk.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path does not exist (when must_exist is True)
            or escapes the project root.
    """
    if not field_path.is_absolute():
        field_path = (manifest_dir / field_path).resolve()
    resolved = field_path.resolve()

    if must_exist and not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")

    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"{label} escapes project root: {resolved} is not under {root}") from None

    return resolved


def load_manifest(path: Path, *, project_root: Path | None = None) -> EvalManifest:
    """Load and validate an evaluation manifest from a YAML file.

    Resolves relative paths against the manifest's parent directory and
    validates that all referenced paths exist and do not escape the project root.

    Args:
        path: Path to the manifest YAML file.
        project_root: Root directory for path traversal validation.
            Defaults to the manifest file's parent directory.

    Returns:
        A validated EvalManifest instance.

    Raises:
        ValueError: If the YAML content is not a mapping, referenced paths
            do not exist, or paths escape the project root.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Manifest must contain a YAML mapping")

    manifest = EvalManifest.model_validate(raw)
    manifest_dir = path.parent.resolve()
    root = project_root.resolve() if project_root is not None else manifest_dir

    manifest.sessions.path = _resolve_and_guard(
        manifest.sessions.path, manifest_dir, root, label="Sessions path"
    )

    for source_index, source in enumerate(manifest.sources):
        source.path = _resolve_and_guard(
            source.path, manifest_dir, root, label=f"sources[{source_index}].path"
        )

    if manifest.ground_truth.path is not None:
        manifest.ground_truth.path = _resolve_and_guard(
            manifest.ground_truth.path, manifest_dir, root, label="ground_truth.path"
        )

    if manifest.synthetic.output is not None:
        manifest.synthetic.output = _resolve_and_guard(
            manifest.synthetic.output,
            manifest_dir,
            root,
            label="synthetic.output",
            must_exist=False,
        )

    return manifest


def discover_manifest() -> Path | None:
    """Discover a manifest file in the current working directory.

    Checks for raki.yaml first, then eval-manifest.yaml.

    Returns:
        Path to the discovered manifest file, or None if not found.
    """
    for name in ["raki.yaml", "eval-manifest.yaml"]:
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate
    return None
