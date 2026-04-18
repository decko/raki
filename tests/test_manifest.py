"""Tests for EvalManifest model and manifest loader."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError


class TestLoadManifest:
    """Tests for load_manifest() function."""

    def test_load_manifest_basic(self, fixtures_dir: Path) -> None:
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(
            fixtures_dir / "manifests" / "basic.yaml",
            project_root=fixtures_dir,
        )
        assert manifest.sessions.path is not None
        assert manifest.sessions.format == "auto"

    def test_load_manifest_validates_sessions_path(self, tmp_path: Path) -> None:
        from raki.ground_truth.manifest import load_manifest

        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text("sessions:\n  path: /nonexistent/path\n  format: auto\n")
        with pytest.raises(ValueError, match="does not exist"):
            load_manifest(manifest_path)

    def test_load_manifest_filter_min_phases(self, fixtures_dir: Path) -> None:
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(
            fixtures_dir / "manifests" / "basic.yaml",
            project_root=fixtures_dir,
        )
        assert manifest.sessions.filter.min_phases == 2

    def test_load_manifest_filter_defaults(self, tmp_path: Path) -> None:
        """Filter defaults should be sensible when not specified."""
        from raki.ground_truth.manifest import load_manifest

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text(f"sessions:\n  path: {sessions_dir}\n  format: auto\n")
        manifest = load_manifest(manifest_path)
        assert manifest.sessions.filter.min_phases == 0
        assert manifest.sessions.filter.tickets is None
        assert manifest.sessions.filter.after is None

    def test_load_manifest_resolves_relative_paths(self, fixtures_dir: Path) -> None:
        """Relative paths should be resolved against manifest parent directory."""
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(
            fixtures_dir / "manifests" / "basic.yaml",
            project_root=fixtures_dir,
        )
        # The path ../sessions relative to manifests/ should resolve to fixtures/sessions
        expected = (fixtures_dir / "sessions").resolve()
        assert manifest.sessions.path == expected

    def test_load_manifest_path_traversal_guard(self, tmp_path: Path) -> None:
        """Paths escaping the project root must be rejected."""
        from raki.ground_truth.manifest import load_manifest

        # Create directory structure: project_root/sub/ contains the manifest
        project_root = tmp_path / "project"
        project_root.mkdir()
        sub_dir = project_root / "sub"
        sub_dir.mkdir()
        # Create sessions directory OUTSIDE the project root
        external_sessions = tmp_path / "external_sessions"
        external_sessions.mkdir()
        manifest_path = sub_dir / "raki.yaml"
        # ../../external_sessions goes: sub/ -> project/ -> tmp_path/external_sessions
        manifest_path.write_text("sessions:\n  path: ../../external_sessions\n  format: auto\n")
        with pytest.raises(ValueError, match="escapes project root"):
            load_manifest(manifest_path, project_root=project_root)

    def test_load_manifest_absolute_path_outside_root(self, tmp_path: Path) -> None:
        """Absolute paths outside project root must be rejected."""
        from raki.ground_truth.manifest import load_manifest

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        manifest_path = project_dir / "raki.yaml"
        manifest_path.write_text(f"sessions:\n  path: {external_dir}\n  format: auto\n")
        with pytest.raises(ValueError, match="escapes project root"):
            load_manifest(manifest_path)

    def test_load_manifest_project_root_defaults_to_manifest_parent(self, tmp_path: Path) -> None:
        """When project_root is not specified, it defaults to the manifest's parent."""
        from raki.ground_truth.manifest import load_manifest

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text("sessions:\n  path: sessions\n  format: auto\n")
        manifest = load_manifest(manifest_path)
        assert manifest.sessions.path == sessions_dir.resolve()

    def test_load_manifest_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML file should raise a clear ValueError."""
        from raki.ground_truth.manifest import load_manifest

        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text("")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_manifest(manifest_path)

    def test_load_manifest_yaml_list_content(self, tmp_path: Path) -> None:
        """YAML with a list at the top level should raise a clear ValueError."""
        from raki.ground_truth.manifest import load_manifest

        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_manifest(manifest_path)

    def test_load_manifest_missing_sessions_key(self, tmp_path: Path) -> None:
        """YAML missing required sessions key should raise ValidationError."""
        from raki.ground_truth.manifest import load_manifest

        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text("ground_truth:\n  path: null\n")
        with pytest.raises(ValidationError, match="sessions"):
            load_manifest(manifest_path)

    def test_load_manifest_source_path_traversal_guard(self, tmp_path: Path) -> None:
        """Source paths escaping the project root must be rejected."""
        from raki.ground_truth.manifest import load_manifest

        project_root = tmp_path / "project"
        project_root.mkdir()
        sessions_dir = project_root / "sessions"
        sessions_dir.mkdir()
        external_docs = tmp_path / "external_docs"
        external_docs.mkdir()
        manifest_path = project_root / "raki.yaml"
        manifest_path.write_text(
            f"sessions:\n  path: sessions\n  format: auto\nsources:\n  - path: {external_docs}\n"
        )
        with pytest.raises(ValueError, match="escapes project root"):
            load_manifest(manifest_path, project_root=project_root)

    def test_load_manifest_ground_truth_path_traversal_guard(self, tmp_path: Path) -> None:
        """Ground truth paths escaping the project root must be rejected."""
        from raki.ground_truth.manifest import load_manifest

        project_root = tmp_path / "project"
        project_root.mkdir()
        sessions_dir = project_root / "sessions"
        sessions_dir.mkdir()
        external_gt = tmp_path / "external_gt"
        external_gt.mkdir()
        manifest_path = project_root / "raki.yaml"
        manifest_path.write_text(
            f"sessions:\n  path: sessions\n  format: auto\nground_truth:\n  path: {external_gt}\n"
        )
        with pytest.raises(ValueError, match="escapes project root"):
            load_manifest(manifest_path, project_root=project_root)

    def test_load_manifest_synthetic_output_traversal_guard(self, tmp_path: Path) -> None:
        """Synthetic output paths escaping the project root must be rejected."""
        from raki.ground_truth.manifest import load_manifest

        project_root = tmp_path / "project"
        project_root.mkdir()
        sessions_dir = project_root / "sessions"
        sessions_dir.mkdir()
        manifest_path = project_root / "raki.yaml"
        manifest_path.write_text(
            "sessions:\n  path: sessions\n  format: auto\n"
            "synthetic:\n  enabled: true\n  output: ../outside\n"
        )
        with pytest.raises(ValueError, match="escapes project root"):
            load_manifest(manifest_path, project_root=project_root)


class TestEvalManifest:
    """Tests for EvalManifest model structure."""

    def test_manifest_has_sessions(self) -> None:
        from raki.ground_truth.manifest import EvalManifest, SessionsConfig

        manifest = EvalManifest(sessions=SessionsConfig(path=Path("/tmp/sessions")))
        assert manifest.sessions.path == Path("/tmp/sessions")

    def test_manifest_sources_default_empty(self) -> None:
        from raki.ground_truth.manifest import EvalManifest, SessionsConfig

        manifest = EvalManifest(sessions=SessionsConfig(path=Path("/tmp/sessions")))
        assert manifest.sources == []

    def test_manifest_ground_truth_default(self) -> None:
        from raki.ground_truth.manifest import EvalManifest, SessionsConfig

        manifest = EvalManifest(sessions=SessionsConfig(path=Path("/tmp/sessions")))
        assert manifest.ground_truth.path is None

    def test_manifest_synthetic_default(self) -> None:
        from raki.ground_truth.manifest import EvalManifest, SessionsConfig

        manifest = EvalManifest(sessions=SessionsConfig(path=Path("/tmp/sessions")))
        assert manifest.synthetic.enabled is False
        assert manifest.synthetic.count == 50
        assert manifest.synthetic.seed == 42


class TestSessionFilter:
    """Tests for SessionFilter model."""

    def test_filter_tickets(self) -> None:
        from raki.ground_truth.manifest import SessionFilter

        session_filter = SessionFilter(tickets=["PROJ-123", "PROJ-456"])
        assert session_filter.tickets == ["PROJ-123", "PROJ-456"]

    def test_filter_after_date(self) -> None:
        from raki.ground_truth.manifest import SessionFilter

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        session_filter = SessionFilter(after=cutoff)
        assert session_filter.after == cutoff

    def test_filter_min_phases(self) -> None:
        from raki.ground_truth.manifest import SessionFilter

        session_filter = SessionFilter(min_phases=3)
        assert session_filter.min_phases == 3


class TestDiscoverManifest:
    """Tests for discover_manifest() function."""

    def test_discover_manifest_raki_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        raki_yaml = tmp_path / "raki.yaml"
        raki_yaml.write_text(f"sessions:\n  path: {sessions_dir}\n  format: auto\n")
        from raki.ground_truth.manifest import discover_manifest

        found = discover_manifest()
        assert found == raki_yaml

    def test_discover_manifest_eval_manifest_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        eval_yaml = tmp_path / "eval-manifest.yaml"
        eval_yaml.write_text("sessions:\n  path: ./sessions\n  format: auto\n")
        from raki.ground_truth.manifest import discover_manifest

        found = discover_manifest()
        assert found == eval_yaml

    def test_discover_manifest_prefers_raki_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both exist, raki.yaml takes priority."""
        monkeypatch.chdir(tmp_path)
        raki_yaml = tmp_path / "raki.yaml"
        raki_yaml.write_text("sessions:\n  path: ./sessions\n  format: auto\n")
        eval_yaml = tmp_path / "eval-manifest.yaml"
        eval_yaml.write_text("sessions:\n  path: ./sessions\n  format: auto\n")
        from raki.ground_truth.manifest import discover_manifest

        found = discover_manifest()
        assert found == raki_yaml

    def test_discover_manifest_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from raki.ground_truth.manifest import discover_manifest

        assert discover_manifest() is None
