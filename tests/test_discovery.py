"""Tests for raki.adapters.discovery — discover_sessions()."""

from __future__ import annotations

import json
from pathlib import Path


from raki.adapters import default_registry
from raki.adapters.discovery import discover_sessions
from raki.adapters.registry import AdapterRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_dir(parent: Path, name: str) -> Path:
    """Create a minimal session-schema session directory."""
    session = parent / name
    session.mkdir()
    (session / "meta.json").write_text(
        json.dumps(
            {
                "ticket": name,
                "started_at": "2026-04-10T08:00:00Z",
                "total_cost": 1.0,
                "rework_cycles": 0,
                "phases": {},
            }
        )
    )
    (session / "events.jsonl").write_text("")
    return session


def _make_alcove_file(parent: Path, name: str = "session.json") -> Path:
    """Create a minimal alcove-format JSON file."""
    path = parent / name
    path.write_text(
        json.dumps(
            {
                "session_id": name.replace(".json", ""),
                "transcript": [
                    {
                        "type": "user",
                        "timestamp": "2026-04-10T08:00:00Z",
                        "message": {"content": "hello"},
                    }
                ],
            }
        )
    )
    return path


# ---------------------------------------------------------------------------
# Basic session detection
# ---------------------------------------------------------------------------


class TestDiscoverSessionsBasic:
    def test_empty_paths_returns_empty(self, tmp_path: Path) -> None:
        registry = default_registry()
        result = discover_sessions([], registry)
        assert result == []

    def test_finds_single_session_schema_dir(self, tmp_path: Path) -> None:
        session = _make_session_dir(tmp_path, "101")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert session.resolve() in result

    def test_finds_multiple_session_schema_dirs(self, tmp_path: Path) -> None:
        sessions = [_make_session_dir(tmp_path, str(i)) for i in range(3)]
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert len(result) == 3
        resolved = {p.resolve() for p in sessions}
        assert set(result) == resolved

    def test_finds_alcove_file_in_directory(self, tmp_path: Path) -> None:
        alcove = _make_alcove_file(tmp_path, "session.json")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert alcove.resolve() in result

    def test_finds_alcove_file_given_directly(self, tmp_path: Path) -> None:
        alcove = _make_alcove_file(tmp_path, "s.json")
        registry = default_registry()
        result = discover_sessions([alcove], registry)
        assert alcove.resolve() in result

    def test_non_session_directory_not_returned(self, tmp_path: Path) -> None:
        # Just an empty directory — no adapter should detect it
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert result == []

    def test_non_json_file_not_returned(self, tmp_path: Path) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert result == []


# ---------------------------------------------------------------------------
# Recursive discovery
# ---------------------------------------------------------------------------


class TestDiscoverSessionsRecursive:
    def test_recursive_finds_nested_sessions(self, tmp_path: Path) -> None:
        """Sessions in subdirectories are discovered recursively (default)."""
        sub = tmp_path / "group-a"
        sub.mkdir()
        session = _make_session_dir(sub, "nested-101")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert session.resolve() in result

    def test_non_recursive_does_not_descend(self, tmp_path: Path) -> None:
        """When recursive=False, subdirectories are not entered."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        _make_session_dir(sub, "101")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry, recursive=False)
        assert result == []

    def test_session_dir_not_recursed_into(self, tmp_path: Path) -> None:
        """A detected session directory is added once; its children are not checked."""
        session = _make_session_dir(tmp_path, "parent-session")
        # Add a nested directory that would also look like a session
        nested = session / "sub-session"
        nested.mkdir()
        (nested / "meta.json").write_text(
            json.dumps({"started_at": "2026-01-01T00:00:00Z", "phases": {}})
        )
        (nested / "events.jsonl").write_text("")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        # Only the top-level session should be returned
        assert len(result) == 1
        assert result[0] == session.resolve()


# ---------------------------------------------------------------------------
# Deduplication and ordering
# ---------------------------------------------------------------------------


class TestDiscoverSessionsDedup:
    def test_no_duplicates_when_path_overlaps(self, tmp_path: Path) -> None:
        """Passing the same path twice should not yield duplicates."""
        _make_session_dir(tmp_path, "101")
        registry = default_registry()
        result = discover_sessions([tmp_path, tmp_path], registry)
        assert len(result) == 1

    def test_no_duplicates_for_same_session_different_paths(self, tmp_path: Path) -> None:
        """A session returned by a parent path and given directly should appear once."""
        session = _make_session_dir(tmp_path, "101")
        registry = default_registry()
        result = discover_sessions([tmp_path, session], registry)
        assert len(result) == 1

    def test_results_are_resolved_paths(self, tmp_path: Path) -> None:
        """All returned paths are absolute (resolved)."""
        _make_session_dir(tmp_path, "101")
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        assert all(p.is_absolute() for p in result)


# ---------------------------------------------------------------------------
# Symlink safety
# ---------------------------------------------------------------------------


class TestDiscoverSessionsSymlinks:
    def test_symlinked_input_path_skipped(self, tmp_path: Path) -> None:
        """A symlinked input path should be silently skipped."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        _make_session_dir(real_dir, "101")
        link = tmp_path / "link"
        link.symlink_to(real_dir)
        registry = default_registry()
        result = discover_sessions([link], registry)
        assert result == []

    def test_symlinked_child_skipped_during_walk(self, tmp_path: Path) -> None:
        """Symlinked children inside a scanned directory are skipped."""
        real_session = _make_session_dir(tmp_path, "real-session")
        link = tmp_path / "link-session"
        link.symlink_to(real_session)
        registry = default_registry()
        result = discover_sessions([tmp_path], registry)
        # Only the real session should appear, not the symlink
        assert len(result) == 1
        assert result[0] == real_session.resolve()


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------


class TestDiscoverSessionsEmptyRegistry:
    def test_empty_registry_finds_nothing(self, tmp_path: Path) -> None:
        """An empty registry has no adapters, so nothing should be detected."""
        _make_session_dir(tmp_path, "101")
        empty_registry = AdapterRegistry()
        result = discover_sessions([tmp_path], empty_registry)
        assert result == []


# ---------------------------------------------------------------------------
# Fixture-based integration
# ---------------------------------------------------------------------------


class TestDiscoverSessionsFixtures:
    def test_discovers_pass_simple_fixture(self, sessions_dir: Path) -> None:
        """discover_sessions finds the pass-simple fixture session."""
        registry = default_registry()
        result = discover_sessions([sessions_dir], registry)
        names = [p.name for p in result]
        assert "pass-simple" in names

    def test_discovers_rework_cycle_fixture(self, sessions_dir: Path) -> None:
        """discover_sessions finds the rework-cycle fixture session."""
        registry = default_registry()
        result = discover_sessions([sessions_dir], registry)
        names = [p.name for p in result]
        assert "rework-cycle" in names

    def test_discovers_all_fixture_sessions(self, sessions_dir: Path) -> None:
        """discover_sessions finds all fixture sessions (at least 3)."""
        registry = default_registry()
        result = discover_sessions([sessions_dir], registry)
        assert len(result) >= 3
