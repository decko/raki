"""Tests for JSONL history log — append-per-run, load, round-trip."""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from raki.model.report import EvalReport
from raki.report.history import (
    HistoryEntry,
    append_history_entry,
    import_history_entry,
    load_history,
    load_run_ids,
)


# ---------------------------------------------------------------------------
# HistoryEntry model
# ---------------------------------------------------------------------------


class TestHistoryEntry:
    def test_fields_present(self) -> None:
        """HistoryEntry must expose run_id, timestamp, sessions_count, metrics."""
        entry = HistoryEntry(
            run_id="eval-001",
            timestamp=datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc),
            sessions_count=42,
            metrics={"first_pass_success_rate": 0.85, "rework_cycles": 1.2},
        )
        assert entry.run_id == "eval-001"
        assert entry.sessions_count == 42
        assert entry.metrics["first_pass_success_rate"] == 0.85

    def test_manifest_optional(self) -> None:
        """manifest must default to None."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sessions_count=1,
            metrics={},
        )
        assert entry.manifest is None

    def test_schema_version_defaults_to_1(self) -> None:
        """schema_version must default to 1."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sessions_count=1,
            metrics={},
        )
        assert entry.schema_version == 1

    def test_config_hash_present(self) -> None:
        """config_hash must be present in the entry."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sessions_count=1,
            metrics={},
            config_hash="abc123",
        )
        assert entry.config_hash == "abc123"

    def test_git_sha_optional(self) -> None:
        """git_sha must default to None."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sessions_count=1,
            metrics={},
        )
        assert entry.git_sha is None

    def test_serializes_to_json(self) -> None:
        """HistoryEntry must serialise to a JSON-compatible dict without errors."""
        entry = HistoryEntry(
            run_id="eval-002",
            timestamp=datetime(2026, 4, 24, 9, 0, 0, tzinfo=timezone.utc),
            sessions_count=10,
            metrics={"cost_efficiency": 7.42},
            manifest="raki.yaml",
        )
        data = entry.model_dump(mode="json")
        assert data["run_id"] == "eval-002"
        assert data["sessions_count"] == 10
        assert data["metrics"]["cost_efficiency"] == 7.42
        assert data["manifest"] == "raki.yaml"
        assert data["schema_version"] == 1

    def test_none_scores_excluded_from_metrics(self) -> None:
        """None values are filtered before storage — metrics dict contains only real values."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sessions_count=5,
            metrics={"rework_cycles": 1.0},
        )
        data = entry.model_dump(mode="json")
        assert "context_precision" not in data["metrics"]
        assert data["metrics"]["rework_cycles"] == 1.0


# ---------------------------------------------------------------------------
# append_history_entry
# ---------------------------------------------------------------------------


def _make_report(
    run_id: str = "eval-test",
    sessions_count: int = 10,
    scores: dict | None = None,
) -> tuple[EvalReport, int]:
    report = EvalReport(
        run_id=run_id,
        timestamp=datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc),
        aggregate_scores=scores or {"first_pass_success_rate": 0.80},
    )
    return report, sessions_count


class TestAppendHistoryEntry:
    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        """append_history_entry must create the JSONL file on first call."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        assert not history_path.exists()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        assert history_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """append_history_entry must create parent directories as needed."""
        history_path = tmp_path / "results" / "deep" / "history.jsonl"
        report, sessions_count = _make_report()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        assert history_path.exists()

    def test_appends_one_line_per_call(self, tmp_path: Path) -> None:
        """Each call must append exactly one JSONL line."""
        history_path = tmp_path / "history.jsonl"
        report_a, count_a = _make_report(run_id="run-1", sessions_count=5)
        report_b, count_b = _make_report(run_id="run-2", sessions_count=8)
        append_history_entry(report_a, history_path, sessions_count=count_a)
        append_history_entry(report_b, history_path, sessions_count=count_b)
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        """Each line in the JSONL file must be valid JSON."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        line = history_path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_written_entry_contains_required_fields(self, tmp_path: Path) -> None:
        """Each JSONL entry must include run_id, timestamp, sessions_count, metrics."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report(run_id="eval-fields", sessions_count=7)
        append_history_entry(report, history_path, sessions_count=sessions_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["run_id"] == "eval-fields"
        assert parsed["sessions_count"] == 7
        assert "timestamp" in parsed
        assert "metrics" in parsed
        assert "schema_version" in parsed
        assert parsed["schema_version"] == 1

    def test_written_entry_carries_manifest_basename(self, tmp_path: Path) -> None:
        """manifest basename from the manifest_file must be written into the history entry."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        manifest_file = tmp_path / "raki.yaml"
        manifest_file.write_text("")
        append_history_entry(
            report, history_path, sessions_count=sessions_count, manifest_file=manifest_file
        )
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["manifest"] == "raki.yaml"

    def test_manifest_none_when_not_provided(self, tmp_path: Path) -> None:
        """manifest must be None when no manifest_file is provided."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["manifest"] is None

    def test_none_scores_excluded_from_metrics(self, tmp_path: Path) -> None:
        """None aggregate scores must be excluded from metrics dict."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report(
            scores={"context_precision": None, "rework_cycles": 1.5}
        )
        append_history_entry(report, history_path, sessions_count=sessions_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert "context_precision" not in parsed["metrics"]
        assert parsed["metrics"]["rework_cycles"] == 1.5

    def test_config_hash_present(self, tmp_path: Path) -> None:
        """config_hash must be present in the JSONL entry."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert "config_hash" in parsed
        assert isinstance(parsed["config_hash"], str)
        assert len(parsed["config_hash"]) == 64  # SHA-256 hex digest

    def test_rejects_symlink(self, tmp_path: Path) -> None:
        """append_history_entry must refuse to write through a symlink."""
        real_file = tmp_path / "real.jsonl"
        real_file.write_text("")
        link_file = tmp_path / "link.jsonl"
        link_file.symlink_to(real_file)
        report, sessions_count = _make_report()
        with pytest.raises(ValueError, match="symlink"):
            append_history_entry(report, link_file, sessions_count=sessions_count)

    def test_second_call_does_not_overwrite_first(self, tmp_path: Path) -> None:
        """Second append must not overwrite the first entry."""
        history_path = tmp_path / "history.jsonl"
        report_a, count_a = _make_report(run_id="run-first")
        report_b, count_b = _make_report(run_id="run-second")
        append_history_entry(report_a, history_path, sessions_count=count_a)
        append_history_entry(report_b, history_path, sessions_count=count_b)
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["run_id"] == "run-first"
        assert second["run_id"] == "run-second"


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------


class TestLoadHistory:
    def test_returns_empty_list_when_file_missing(self, tmp_path: Path) -> None:
        """load_history must return [] when the file does not exist."""
        history_path = tmp_path / "nonexistent.jsonl"
        entries = load_history(history_path)
        assert entries == []

    def test_round_trip_single_entry(self, tmp_path: Path) -> None:
        """A single entry written then read must match the original report."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report(run_id="round-trip", sessions_count=15)
        report.aggregate_scores = {"rework_cycles": 2.1}
        append_history_entry(report, history_path, sessions_count=sessions_count)
        entries = load_history(history_path)
        assert len(entries) == 1
        assert entries[0].run_id == "round-trip"
        assert entries[0].sessions_count == 15
        assert entries[0].metrics["rework_cycles"] == 2.1

    def test_round_trip_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple appended entries must all be loaded in order."""
        history_path = tmp_path / "history.jsonl"
        for idx in range(3):
            report, sessions_count = _make_report(run_id=f"run-{idx}", sessions_count=idx + 1)
            append_history_entry(report, history_path, sessions_count=sessions_count)
        entries = load_history(history_path)
        assert len(entries) == 3
        assert entries[0].run_id == "run-0"
        assert entries[2].run_id == "run-2"

    def test_entries_are_history_entry_instances(self, tmp_path: Path) -> None:
        """load_history must return HistoryEntry instances, not raw dicts."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        entries = load_history(history_path)
        assert isinstance(entries[0], HistoryEntry)

    def test_rejects_symlink(self, tmp_path: Path) -> None:
        """load_history must refuse to read through a symlink."""
        real_file = tmp_path / "real.jsonl"
        real_file.write_text(
            '{"run_id":"r","timestamp":"2026-01-01T00:00:00Z","sessions_count":1,"metrics":{}}\n'
        )
        link_file = tmp_path / "link.jsonl"
        link_file.symlink_to(real_file)
        with pytest.raises(ValueError, match="symlink"):
            load_history(link_file)

    def test_metrics_round_trip(self, tmp_path: Path) -> None:
        """Metrics must survive JSONL round-trip correctly."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report(scores={"rework_cycles": 1.5})
        append_history_entry(report, history_path, sessions_count=sessions_count)
        entries = load_history(history_path)
        assert entries[0].metrics["rework_cycles"] == 1.5

    def test_timestamp_timezone_preserved(self, tmp_path: Path) -> None:
        """Timestamp timezone info must be preserved after JSONL round-trip."""
        history_path = tmp_path / "history.jsonl"
        report, sessions_count = _make_report()
        append_history_entry(report, history_path, sessions_count=sessions_count)
        entries = load_history(history_path)
        assert entries[0].timestamp.tzinfo is not None

    def test_malformed_lines_skipped_with_warning(self, tmp_path: Path) -> None:
        """Malformed JSONL lines must be skipped with a warning, not raise."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            '{"run_id":"good","timestamp":"2026-01-01T00:00:00Z","sessions_count":1,"metrics":{}}\n'
            "not valid json at all\n"
            '{"run_id":"also-good","timestamp":"2026-02-01T00:00:00Z","sessions_count":2,"metrics":{}}\n'
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entries = load_history(history_path)
        assert len(entries) == 2
        assert entries[0].run_id == "good"
        assert entries[1].run_id == "also-good"
        assert len(caught) == 1
        assert "malformed" in str(caught[0].message).lower() or "Skipping" in str(caught[0].message)

    def test_validation_error_lines_skipped_with_warning(self, tmp_path: Path) -> None:
        """Lines with valid JSON but invalid schema must be skipped with a warning."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            '{"run_id":"good","timestamp":"2026-01-01T00:00:00Z","sessions_count":1,"metrics":{}}\n'
            '{"completely": "wrong schema"}\n'
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entries = load_history(history_path)
        assert len(entries) == 1
        assert entries[0].run_id == "good"
        assert len(caught) == 1

    def test_git_sha_populated_in_git_repo(self, tmp_path: Path) -> None:
        """git_sha must be populated when running inside a git repo."""
        from raki.report.history import _git_sha

        # We are running tests inside a git repo, so _git_sha() should return a value
        sha = _git_sha()
        # In CI or repo context, this should be a short hex string
        assert sha is not None
        assert len(sha) >= 7
        assert all(char in "0123456789abcdef" for char in sha)

    def test_git_sha_none_outside_git_repo(self, tmp_path: Path) -> None:
        """git_sha must be None when not in a git repo."""
        from raki.report.history import _git_sha

        with patch(
            "raki.report.history.subprocess.run",
            return_value=type("Result", (), {"returncode": 128, "stdout": ""})(),
        ):
            sha = _git_sha()
        assert sha is None


# ---------------------------------------------------------------------------
# load_run_ids
# ---------------------------------------------------------------------------


class TestLoadRunIds:
    def test_returns_empty_set_when_file_missing(self, tmp_path: Path) -> None:
        """load_run_ids must return an empty set when the history file does not exist."""
        history_path = tmp_path / "history.jsonl"
        ids = load_run_ids(history_path)
        assert ids == set()

    def test_returns_all_run_ids(self, tmp_path: Path) -> None:
        """load_run_ids must return all run_id values from the history file."""
        history_path = tmp_path / "history.jsonl"
        for run_id in ("eval-a", "eval-b", "eval-c"):
            report, count = _make_report(run_id=run_id)
            append_history_entry(report, history_path, sessions_count=count)
        ids = load_run_ids(history_path)
        assert ids == {"eval-a", "eval-b", "eval-c"}

    def test_returns_set_not_list(self, tmp_path: Path) -> None:
        """load_run_ids must return a set for O(1) membership tests."""
        history_path = tmp_path / "history.jsonl"
        report, count = _make_report()
        append_history_entry(report, history_path, sessions_count=count)
        result = load_run_ids(history_path)
        assert isinstance(result, set)

    def test_rejects_symlink(self, tmp_path: Path) -> None:
        """load_run_ids must refuse to read a symlink (delegates to load_history)."""
        real_file = tmp_path / "real.jsonl"
        real_file.write_text(
            '{"run_id":"r","timestamp":"2026-01-01T00:00:00Z","sessions_count":1,"metrics":{}}\n'
        )
        link = tmp_path / "link.jsonl"
        link.symlink_to(real_file)
        with pytest.raises(ValueError, match="symlink"):
            load_run_ids(link)


# ---------------------------------------------------------------------------
# import_history_entry
# ---------------------------------------------------------------------------


def _make_entry(run_id: str = "import-101") -> HistoryEntry:
    return HistoryEntry(
        run_id=run_id,
        timestamp=datetime(2026, 4, 10, 8, 0, 0, tzinfo=timezone.utc),
        sessions_count=1,
        metrics={"first_pass_success_rate": 1.0},
    )


class TestImportHistoryEntry:
    def test_writes_new_entry(self, tmp_path: Path) -> None:
        """import_history_entry must write a new entry and return True."""
        history_path = tmp_path / "history.jsonl"
        entry = _make_entry("import-101")
        existing: set[str] = set()
        result = import_history_entry(entry, history_path, existing)
        assert result is True
        assert history_path.exists()

    def test_skips_duplicate_run_id(self, tmp_path: Path) -> None:
        """import_history_entry must skip an entry whose run_id is already present."""
        history_path = tmp_path / "history.jsonl"
        entry = _make_entry("import-101")
        existing: set[str] = {"import-101"}
        result = import_history_entry(entry, history_path, existing)
        assert result is False
        assert not history_path.exists()

    def test_updates_existing_ids_on_write(self, tmp_path: Path) -> None:
        """import_history_entry must add the run_id to existing_ids after writing."""
        history_path = tmp_path / "history.jsonl"
        entry = _make_entry("import-101")
        existing: set[str] = set()
        import_history_entry(entry, history_path, existing)
        assert "import-101" in existing

    def test_does_not_update_existing_ids_on_skip(self, tmp_path: Path) -> None:
        """existing_ids must not be modified when the entry is skipped."""
        history_path = tmp_path / "history.jsonl"
        entry = _make_entry("import-101")
        existing: set[str] = {"import-101"}
        import_history_entry(entry, history_path, existing)
        assert existing == {"import-101"}

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple import_history_entry calls must produce multiple JSONL lines."""
        history_path = tmp_path / "history.jsonl"
        existing: set[str] = set()
        for i in range(3):
            entry = _make_entry(f"import-{i}")
            import_history_entry(entry, history_path, existing)
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_written_line_is_valid_json(self, tmp_path: Path) -> None:
        """Each written line must be valid JSON."""
        history_path = tmp_path / "history.jsonl"
        entry = _make_entry("import-json")
        import_history_entry(entry, history_path, set())
        raw = history_path.read_text(encoding="utf-8").strip()
        parsed = json.loads(raw)
        assert parsed["run_id"] == "import-json"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """import_history_entry must create missing parent directories."""
        history_path = tmp_path / "deep" / "nested" / "history.jsonl"
        entry = _make_entry("import-deep")
        import_history_entry(entry, history_path, set())
        assert history_path.exists()

    def test_rejects_symlink(self, tmp_path: Path) -> None:
        """import_history_entry must refuse to write through a symlink."""
        real_file = tmp_path / "real.jsonl"
        real_file.write_text("")
        link = tmp_path / "link.jsonl"
        link.symlink_to(real_file)
        entry = _make_entry("import-sym")
        with pytest.raises(ValueError, match="symlink"):
            import_history_entry(entry, link, set())

    def test_second_call_same_id_is_idempotent(self, tmp_path: Path) -> None:
        """After a successful write, calling import_history_entry again with the
        same run_id must be a no-op (existing_ids is updated in-place)."""
        history_path = tmp_path / "history.jsonl"
        entry = _make_entry("import-idem")
        existing: set[str] = set()
        import_history_entry(entry, history_path, existing)
        result = import_history_entry(entry, history_path, existing)
        assert result is False
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
