"""Tests for the `raki import-history` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from raki.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_dir(parent: Path, name: str = "101") -> Path:
    """Create a minimal session-schema session directory."""
    session = parent / name
    session.mkdir()
    (session / "meta.json").write_text(
        json.dumps(
            {
                "ticket": name,
                "started_at": "2026-04-10T08:00:00Z",
                "total_cost": 5.0,
                "rework_cycles": 0,
                "phases": {
                    "triage": {"status": "completed", "generation": 1},
                    "implement": {"status": "completed", "generation": 1},
                    "verify": {"status": "completed", "generation": 1},
                },
            }
        )
    )
    (session / "events.jsonl").write_text("")
    (session / "verify.json").write_text(json.dumps({"verdict": "PASS"}))
    return session


def _make_session_dir_with_rework(parent: Path, name: str = "102") -> Path:
    """Create a session-schema directory with 1 rework cycle."""
    session = parent / name
    session.mkdir()
    (session / "meta.json").write_text(
        json.dumps(
            {
                "ticket": name,
                "started_at": "2026-04-11T08:00:00Z",
                "total_cost": 8.0,
                "rework_cycles": 1,
                "phases": {
                    "implement": {"status": "completed", "generation": 2},
                    "verify": {"status": "completed", "generation": 1},
                },
            }
        )
    )
    (session / "events.jsonl").write_text("")
    (session / "verify.json").write_text(json.dumps({"verdict": "PASS"}))
    return session


def _invoke(args: list[str], *, cwd: Path | None = None) -> object:
    """Invoke the CLI from within *cwd* (or tmp_path via monkeypatch)."""
    runner = CliRunner()
    return runner.invoke(main, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Help and registration
# ---------------------------------------------------------------------------


class TestImportHistoryHelp:
    def test_import_history_in_help(self) -> None:
        """import-history must appear in the top-level help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "import-history" in result.output

    def test_import_history_help_text(self) -> None:
        """import-history --help must show usage, options, and description."""
        runner = CliRunner()
        result = runner.invoke(main, ["import-history", "--help"])
        assert result.exit_code == 0
        assert "PATHS" in result.output
        assert "--dry-run" in result.output
        assert "--history-path" in result.output
        assert "--adapter" in result.output
        assert "-q" in result.output or "--quiet" in result.output


# ---------------------------------------------------------------------------
# Basic import
# ---------------------------------------------------------------------------


class TestImportHistoryBasic:
    def test_imports_single_session(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """import-history must import a single detected session."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert history_path.exists()
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_imports_multiple_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """import-history must import all detected sessions in the directory."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        for name in ("101", "102", "103"):
            _make_session_dir(sessions, name)
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_each_entry_is_valid_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each line written to history.jsonl must be valid JSON."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        for line in history_path.read_text(encoding="utf-8").strip().splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            assert "run_id" in parsed
            assert "timestamp" in parsed
            assert "sessions_count" in parsed
            assert "metrics" in parsed

    def test_run_id_prefixed_with_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Imported entries must have run_id starting with 'import-'."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["run_id"].startswith("import-")

    def test_timestamp_from_session_started_at(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Imported entry timestamp must come from the session's started_at field."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert "2026-04-10" in parsed["timestamp"]

    def test_sessions_count_is_one_per_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each imported entry must have sessions_count == 1."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        _make_session_dir(sessions, "102")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        for line in history_path.read_text(encoding="utf-8").strip().splitlines():
            parsed = json.loads(line)
            assert parsed["sessions_count"] == 1

    def test_metrics_are_computed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Imported entries must contain at least one operational metric."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert len(parsed["metrics"]) > 0

    def test_output_reports_imported_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The output summary must report how many sessions were imported."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        _make_session_dir(sessions, "102")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert "2" in result.output
        assert "import" in result.output.lower()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestImportHistoryDeduplication:
    def test_skips_already_imported_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second import of the same session must be skipped."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_new_sessions_appended_to_existing_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New sessions are added while existing entries are preserved."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        _make_session_dir(sessions, "102")
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_duplicate_report_shows_already_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Output on second import must mention the session is already present."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        result = runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert "already" in result.output.lower() or "present" in result.output.lower()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestImportHistoryDryRun:
    def test_dry_run_does_not_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--dry-run must not create or modify the history file."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            [
                "import-history",
                str(sessions),
                "--history-path",
                str(history_path),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert not history_path.exists()

    def test_dry_run_reports_would_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run output must mention 'would import'."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-history",
                str(sessions),
                "--history-path",
                str(history_path),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "would import" in result.output.lower() or "dry run" in result.output.lower()

    def test_dry_run_shows_session_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run output must include the count of sessions that would be imported."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        for name in ("101", "102"):
            _make_session_dir(sessions, name)
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-history",
                str(sessions),
                "--history-path",
                str(history_path),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert "2" in result.output


# ---------------------------------------------------------------------------
# Quiet mode
# ---------------------------------------------------------------------------


class TestImportHistoryQuiet:
    def test_quiet_suppresses_per_session_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--quiet must suppress per-session lines but still write the history."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-history",
                str(sessions),
                "--history-path",
                str(history_path),
                "-q",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert history_path.exists()
        # Per-session lines (✓ / ✗) must not appear
        assert "✓" not in result.output
        assert "✗" not in result.output


# ---------------------------------------------------------------------------
# No sessions found
# ---------------------------------------------------------------------------


class TestImportHistoryNoSessions:
    def test_empty_directory_reports_no_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty directory must produce a 'no sessions' message."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "no sessions" in result.output.lower()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestImportHistoryErrors:
    def test_invalid_adapter_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognised adapter name must produce a clear error and exit."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-history",
                str(sessions),
                "--history-path",
                str(history_path),
                "--adapter",
                "nonexistent-adapter",
            ],
        )
        assert result.exit_code != 0

    def test_malformed_session_logged_as_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session that fails to load must be counted as an error, not cause a crash."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        # Create a malformed session (invalid JSON in meta.json)
        malformed = sessions / "bad-session"
        malformed.mkdir()
        (malformed / "meta.json").write_text("NOT JSON {{{")
        (malformed / "events.jsonl").write_text("")
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-history", str(sessions), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "error" in result.output.lower()

    def test_history_path_outside_project_root_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A history-path outside the project root must be rejected with UsageError."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        _make_session_dir(sessions, "101")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-history",
                str(sessions),
                "--history-path",
                "/tmp/outside-project-history.jsonl",
            ],
        )
        assert result.exit_code != 0
        assert "project root" in result.output.lower() or result.exit_code == 2


# ---------------------------------------------------------------------------
# Fixture-based integration
# ---------------------------------------------------------------------------


class TestImportHistoryFixtures:
    def test_imports_from_fixture_sessions_dir(
        self, sessions_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """import-history must successfully import from the test fixtures directory."""
        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-history", str(sessions_dir), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert history_path.exists()
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3

    def test_imported_entries_have_valid_schema(
        self, sessions_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All imported entries must conform to the HistoryEntry schema."""
        from raki.report.history import HistoryEntry

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions_dir), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        for line in history_path.read_text(encoding="utf-8").strip().splitlines():
            parsed = json.loads(line)
            entry = HistoryEntry.model_validate(parsed)
            assert entry.sessions_count == 1
            assert entry.run_id.startswith("import-")

    def test_repeated_import_is_idempotent(
        self, sessions_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running import-history twice on the same path must not duplicate entries."""
        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"

        runner = CliRunner()
        runner.invoke(
            main,
            ["import-history", str(sessions_dir), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        first_count = len(history_path.read_text(encoding="utf-8").strip().splitlines())

        runner.invoke(
            main,
            ["import-history", str(sessions_dir), "--history-path", str(history_path)],
            catch_exceptions=False,
        )
        second_count = len(history_path.read_text(encoding="utf-8").strip().splitlines())

        assert first_count == second_count
