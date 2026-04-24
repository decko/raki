"""Tests for JSONL history log — append-per-run, load, round-trip."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from raki.model.report import EvalReport
from raki.report.history import HistoryEntry, append_history_entry, load_history


# ---------------------------------------------------------------------------
# HistoryEntry model
# ---------------------------------------------------------------------------


class TestHistoryEntry:
    def test_fields_present(self) -> None:
        """HistoryEntry must expose run_id, timestamp, session_count, aggregate_scores."""
        entry = HistoryEntry(
            run_id="eval-001",
            timestamp=datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc),
            session_count=42,
            aggregate_scores={"first_pass_success_rate": 0.85, "rework_cycles": 1.2},
        )
        assert entry.run_id == "eval-001"
        assert entry.session_count == 42
        assert entry.aggregate_scores["first_pass_success_rate"] == 0.85

    def test_manifest_hash_optional(self) -> None:
        """manifest_hash must default to None."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            session_count=1,
            aggregate_scores={},
        )
        assert entry.manifest_hash is None

    def test_serializes_to_json(self) -> None:
        """HistoryEntry must serialise to a JSON-compatible dict without errors."""
        entry = HistoryEntry(
            run_id="eval-002",
            timestamp=datetime(2026, 4, 24, 9, 0, 0, tzinfo=timezone.utc),
            session_count=10,
            aggregate_scores={"cost_efficiency": 7.42},
            manifest_hash="abc123",
        )
        data = entry.model_dump(mode="json")
        assert data["run_id"] == "eval-002"
        assert data["session_count"] == 10
        assert data["aggregate_scores"]["cost_efficiency"] == 7.42
        assert data["manifest_hash"] == "abc123"

    def test_none_scores_preserved(self) -> None:
        """None aggregate scores (skipped metrics) must survive model_dump."""
        entry = HistoryEntry(
            run_id="r",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            session_count=5,
            aggregate_scores={"context_precision": None, "rework_cycles": 1.0},
        )
        data = entry.model_dump(mode="json")
        assert data["aggregate_scores"]["context_precision"] is None


# ---------------------------------------------------------------------------
# append_history_entry
# ---------------------------------------------------------------------------


def _make_report(
    run_id: str = "eval-test",
    session_count: int = 10,
    scores: dict | None = None,
) -> tuple[EvalReport, int]:
    report = EvalReport(
        run_id=run_id,
        timestamp=datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc),
        aggregate_scores=scores or {"first_pass_success_rate": 0.80},
        manifest_hash="deadbeef",
    )
    return report, session_count


class TestAppendHistoryEntry:
    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        """append_history_entry must create the JSONL file on first call."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report()
        assert not history_path.exists()
        append_history_entry(report, history_path, session_count=session_count)
        assert history_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """append_history_entry must create parent directories as needed."""
        history_path = tmp_path / "results" / "deep" / "raki-history.jsonl"
        report, session_count = _make_report()
        append_history_entry(report, history_path, session_count=session_count)
        assert history_path.exists()

    def test_appends_one_line_per_call(self, tmp_path: Path) -> None:
        """Each call must append exactly one JSONL line."""
        history_path = tmp_path / "raki-history.jsonl"
        report_a, count_a = _make_report(run_id="run-1", session_count=5)
        report_b, count_b = _make_report(run_id="run-2", session_count=8)
        append_history_entry(report_a, history_path, session_count=count_a)
        append_history_entry(report_b, history_path, session_count=count_b)
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        """Each line in the JSONL file must be valid JSON."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report()
        append_history_entry(report, history_path, session_count=session_count)
        line = history_path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_written_entry_contains_required_fields(self, tmp_path: Path) -> None:
        """Each JSONL entry must include run_id, timestamp, session_count, aggregate_scores."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report(run_id="eval-fields", session_count=7)
        append_history_entry(report, history_path, session_count=session_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["run_id"] == "eval-fields"
        assert parsed["session_count"] == 7
        assert "timestamp" in parsed
        assert "aggregate_scores" in parsed

    def test_written_entry_carries_manifest_hash(self, tmp_path: Path) -> None:
        """manifest_hash from the report must be written into the history entry."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report()
        report.manifest_hash = "cafebabe"
        append_history_entry(report, history_path, session_count=session_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["manifest_hash"] == "cafebabe"

    def test_none_score_serialized_as_null(self, tmp_path: Path) -> None:
        """None aggregate scores must appear as JSON null, not 0.0."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report(
            scores={"context_precision": None, "rework_cycles": 1.5}
        )
        append_history_entry(report, history_path, session_count=session_count)
        parsed = json.loads(history_path.read_text(encoding="utf-8").strip())
        assert parsed["aggregate_scores"]["context_precision"] is None

    def test_rejects_symlink(self, tmp_path: Path) -> None:
        """append_history_entry must refuse to write through a symlink."""
        real_file = tmp_path / "real.jsonl"
        real_file.write_text("")
        link_file = tmp_path / "link.jsonl"
        link_file.symlink_to(real_file)
        report, session_count = _make_report()
        with pytest.raises(ValueError, match="symlink"):
            append_history_entry(report, link_file, session_count=session_count)

    def test_second_call_does_not_overwrite_first(self, tmp_path: Path) -> None:
        """Second append must not overwrite the first entry."""
        history_path = tmp_path / "raki-history.jsonl"
        report_a, count_a = _make_report(run_id="run-first")
        report_b, count_b = _make_report(run_id="run-second")
        append_history_entry(report_a, history_path, session_count=count_a)
        append_history_entry(report_b, history_path, session_count=count_b)
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
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report(run_id="round-trip", session_count=15)
        report.aggregate_scores = {"rework_cycles": 2.1}
        append_history_entry(report, history_path, session_count=session_count)
        entries = load_history(history_path)
        assert len(entries) == 1
        assert entries[0].run_id == "round-trip"
        assert entries[0].session_count == 15
        assert entries[0].aggregate_scores["rework_cycles"] == 2.1

    def test_round_trip_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple appended entries must all be loaded in order."""
        history_path = tmp_path / "raki-history.jsonl"
        for idx in range(3):
            report, session_count = _make_report(run_id=f"run-{idx}", session_count=idx + 1)
            append_history_entry(report, history_path, session_count=session_count)
        entries = load_history(history_path)
        assert len(entries) == 3
        assert entries[0].run_id == "run-0"
        assert entries[2].run_id == "run-2"

    def test_entries_are_history_entry_instances(self, tmp_path: Path) -> None:
        """load_history must return HistoryEntry instances, not raw dicts."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report()
        append_history_entry(report, history_path, session_count=session_count)
        entries = load_history(history_path)
        assert isinstance(entries[0], HistoryEntry)

    def test_rejects_symlink(self, tmp_path: Path) -> None:
        """load_history must refuse to read through a symlink."""
        real_file = tmp_path / "real.jsonl"
        real_file.write_text(
            '{"run_id":"r","timestamp":"2026-01-01T00:00:00Z","session_count":1,"aggregate_scores":{}}\n'
        )
        link_file = tmp_path / "link.jsonl"
        link_file.symlink_to(real_file)
        with pytest.raises(ValueError, match="symlink"):
            load_history(link_file)

    def test_none_scores_preserved_after_round_trip(self, tmp_path: Path) -> None:
        """None scores must survive JSONL round-trip as None, not 0.0."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report(
            scores={"context_precision": None, "rework_cycles": 1.5}
        )
        append_history_entry(report, history_path, session_count=session_count)
        entries = load_history(history_path)
        assert entries[0].aggregate_scores["context_precision"] is None
        assert entries[0].aggregate_scores["rework_cycles"] == 1.5

    def test_timestamp_timezone_preserved(self, tmp_path: Path) -> None:
        """Timestamp timezone info must be preserved after JSONL round-trip."""
        history_path = tmp_path / "raki-history.jsonl"
        report, session_count = _make_report()
        append_history_entry(report, history_path, session_count=session_count)
        entries = load_history(history_path)
        assert entries[0].timestamp.tzinfo is not None
