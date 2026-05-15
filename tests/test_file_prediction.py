"""Tests for FilePredictionAccuracyMetric.

Tests that validate how well the agent's triage file predictions match the
actual files changed during implementation.

All tests use local helper factories to avoid polluting conftest.py with
fixtures only relevant to this single metric.
"""

from datetime import datetime, timezone

import pytest

from raki.metrics.operational.file_prediction import (
    FilePredictionAccuracyMetric,
    _extract_actual_files,
    _extract_predicted_files,
    _normalize_path,
)
from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset, EvalSample, PhaseResult, SessionMeta


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _make_meta(session_id: str) -> SessionMeta:
    return SessionMeta(
        session_id=session_id,
        started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        total_phases=2,
        rework_cycles=0,
    )


def _make_phase(
    name: str,
    *,
    output_structured: dict | None = None,
    files_modified: list[str] | None = None,
) -> PhaseResult:
    return PhaseResult(
        name=name,
        generation=1,
        status="completed",
        output="done",
        output_structured=output_structured,
        files_modified=files_modified or [],
    )


def _make_sample(
    session_id: str,
    triage_files: list[str] | None = None,
    actual_files_changed: list[dict] | None = None,
    actual_files_modified: list[str] | None = None,
    include_triage: bool = True,
    include_implement: bool = True,
) -> EvalSample:
    """Build a sample with triage and/or implement phases.

    - triage_files: predicted files in triage.output_structured["files"]
    - actual_files_changed: files_changed list in implement.output_structured (SODA format)
    - actual_files_modified: files_modified list on the PhaseResult (Alcove format)
    """
    phases = []
    if include_triage:
        triage_structured = {}
        if triage_files is not None:
            triage_structured["files"] = triage_files
        phases.append(_make_phase("triage", output_structured=triage_structured or None))
    if include_implement:
        impl_structured = {}
        if actual_files_changed is not None:
            impl_structured["files_changed"] = actual_files_changed
        phases.append(
            _make_phase(
                "implement",
                output_structured=impl_structured or None,
                files_modified=actual_files_modified,
            )
        )
    return EvalSample(session=_make_meta(session_id), phases=phases, findings=[], events=[])


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestNormalizePath:
    """_normalize_path strips leading './', normalises to lowercase."""

    def test_strips_leading_dotslash(self) -> None:
        assert _normalize_path("./src/foo.py") == "src/foo.py"

    def test_lowercases_path(self) -> None:
        assert _normalize_path("Src/Foo.PY") == "src/foo.py"

    def test_strips_dotslash_and_lowercases(self) -> None:
        assert _normalize_path("./Src/Bar.PY") == "src/bar.py"

    def test_already_clean_path_unchanged(self) -> None:
        assert _normalize_path("src/foo.py") == "src/foo.py"

    def test_empty_string_unchanged(self) -> None:
        assert _normalize_path("") == ""


class TestExtractPredictedFiles:
    """_extract_predicted_files reads triage.output_structured['files']."""

    def test_returns_files_from_triage_phase(self) -> None:
        sample = _make_sample(
            "s1",
            triage_files=["src/foo.py", "src/bar.py"],
        )
        result = _extract_predicted_files(sample)
        assert result == {"src/foo.py", "src/bar.py"}

    def test_returns_empty_when_no_triage_phase(self) -> None:
        sample = _make_sample("s2", include_triage=False)
        assert _extract_predicted_files(sample) == set()

    def test_returns_empty_when_files_key_missing(self) -> None:
        sample = _make_sample("s3", triage_files=None)
        # triage phase exists but no 'files' key
        assert _extract_predicted_files(sample) == set()

    def test_normalizes_paths(self) -> None:
        sample = _make_sample("s4", triage_files=["./Src/Foo.PY"])
        result = _extract_predicted_files(sample)
        assert result == {"src/foo.py"}


class TestExtractActualFiles:
    """_extract_actual_files reads implement phase file data."""

    def test_soda_format_files_changed_list_of_dicts(self) -> None:
        """files_changed as list of dicts with 'path' key (SODA format)."""
        sample = _make_sample(
            "s1",
            actual_files_changed=[
                {"path": "src/foo.py", "action": "modified"},
                {"path": "src/bar.py", "action": "created"},
            ],
        )
        result = _extract_actual_files(sample)
        assert result == {"src/foo.py", "src/bar.py"}

    def test_alcove_format_files_modified(self) -> None:
        """files_modified as list of strings on the PhaseResult (Alcove format)."""
        sample = _make_sample(
            "s2",
            actual_files_modified=["src/runner.go", "internal/engine.go"],
        )
        result = _extract_actual_files(sample)
        assert result == {"src/runner.go", "internal/engine.go"}

    def test_returns_empty_when_no_implement_phase(self) -> None:
        sample = _make_sample("s3", include_implement=False)
        assert _extract_actual_files(sample) == set()

    def test_normalizes_paths(self) -> None:
        sample = _make_sample(
            "s4",
            actual_files_changed=[{"path": "./Src/Foo.PY", "action": "modified"}],
        )
        result = _extract_actual_files(sample)
        assert result == {"src/foo.py"}

    def test_prefers_files_changed_over_files_modified(self) -> None:
        """When both files_changed and files_modified are present, prefer files_changed."""
        sample = _make_sample(
            "s5",
            actual_files_changed=[{"path": "src/from_changed.py", "action": "modified"}],
            actual_files_modified=["src/from_modified.py"],
        )
        result = _extract_actual_files(sample)
        assert result == {"src/from_changed.py"}

    def test_falls_back_to_files_modified_when_files_changed_empty(self) -> None:
        """When files_changed is empty, fall back to files_modified."""
        sample = _make_sample(
            "s6",
            actual_files_changed=[],
            actual_files_modified=["src/fallback.py"],
        )
        result = _extract_actual_files(sample)
        assert result == {"src/fallback.py"}


# ---------------------------------------------------------------------------
# FilePredictionAccuracyMetric tests
# ---------------------------------------------------------------------------


class TestFilePredictionAccuracyPerfect:
    """Perfect predictions: predicted set equals actual set."""

    def test_perfect_prediction_scores_one(self) -> None:
        sample = _make_sample(
            "s1",
            triage_files=["src/foo.py", "src/bar.py"],
            actual_files_changed=[
                {"path": "src/foo.py", "action": "modified"},
                {"path": "src/bar.py", "action": "modified"},
            ],
        )
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score == pytest.approx(1.0)
        assert result.sample_scores["s1"] == pytest.approx(1.0)


class TestFilePredictionAccuracyNoOverlap:
    """No overlap between predicted and actual files."""

    def test_no_overlap_scores_zero(self) -> None:
        sample = _make_sample(
            "s2",
            triage_files=["src/foo.py"],
            actual_files_changed=[{"path": "src/bar.py", "action": "modified"}],
        )
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score == pytest.approx(0.0)
        assert result.sample_scores["s2"] == pytest.approx(0.0)


class TestFilePredictionAccuracyPartialOverlap:
    """Partial overlap: precision and recall differ."""

    def test_partial_overlap_f1(self) -> None:
        """Predicted: {A, B}, Actual: {B, C}.
        Intersection: {B} → precision=0.5, recall=0.5, F1=0.5.
        """
        sample = _make_sample(
            "s3",
            triage_files=["src/a.py", "src/b.py"],
            actual_files_changed=[
                {"path": "src/b.py", "action": "modified"},
                {"path": "src/c.py", "action": "created"},
            ],
        )
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score == pytest.approx(0.5)


class TestFilePredictionAccuracyNAConditions:
    """N/A when no sessions have triage file predictions."""

    def test_no_triage_phase_returns_na(self) -> None:
        sample = _make_sample("s4", include_triage=False)
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["sessions_with_file_predictions"] == 0

    def test_empty_triage_files_excluded(self) -> None:
        """Empty triage files list → session excluded → N/A."""
        sample = _make_sample("s5", triage_files=[])
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["sessions_with_file_predictions"] == 0

    def test_empty_dataset_returns_na(self) -> None:
        dataset = EvalDataset(samples=[])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["sessions_with_file_predictions"] == 0

    def test_actual_files_empty_f1_zero(self) -> None:
        """Predicted files exist but no actual files → precision=0, recall=0, F1=0."""
        sample = _make_sample(
            "s6",
            triage_files=["src/foo.py"],
            actual_files_changed=[],
        )
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score == pytest.approx(0.0)
        assert result.sample_scores["s6"] == pytest.approx(0.0)


class TestFilePredictionAccuracyMultipleSessions:
    """Mean F1 across multiple sessions."""

    def test_mean_f1_across_sessions(self) -> None:
        """Two sessions: F1=1.0 and F1=0.0 → mean=0.5."""
        perfect = _make_sample(
            "s-perfect",
            triage_files=["src/foo.py"],
            actual_files_changed=[{"path": "src/foo.py", "action": "modified"}],
        )
        wrong = _make_sample(
            "s-wrong",
            triage_files=["src/foo.py"],
            actual_files_changed=[{"path": "src/bar.py", "action": "modified"}],
        )
        dataset = EvalDataset(samples=[perfect, wrong])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.score == pytest.approx(0.5)
        assert result.sample_scores["s-perfect"] == pytest.approx(1.0)
        assert result.sample_scores["s-wrong"] == pytest.approx(0.0)


class TestFilePredictionAccuracyDetails:
    """Details dict structure."""

    def test_details_include_sessions_with_file_predictions(self) -> None:
        sample = _make_sample(
            "s1",
            triage_files=["src/foo.py"],
            actual_files_changed=[{"path": "src/foo.py", "action": "modified"}],
        )
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert "sessions_with_file_predictions" in result.details
        assert result.details["sessions_with_file_predictions"] == 1

    def test_details_include_micro_average(self) -> None:
        """Details should include micro_precision, micro_recall, micro_f1."""
        sample = _make_sample(
            "s1",
            triage_files=["src/foo.py", "src/bar.py"],
            actual_files_changed=[{"path": "src/foo.py", "action": "modified"}],
        )
        dataset = EvalDataset(samples=[sample])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert "micro_precision" in result.details
        assert "micro_recall" in result.details
        assert "micro_f1" in result.details

    def test_na_details_include_sessions_with_file_predictions_zero(self) -> None:
        """N/A result must expose sessions_with_file_predictions=0."""
        dataset = EvalDataset(samples=[])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.details["sessions_with_file_predictions"] == 0

    def test_sessions_without_predictions_excluded_from_count(self) -> None:
        """Sessions without triage file predictions should not count."""
        qualifying = _make_sample(
            "qualifying",
            triage_files=["src/foo.py"],
            actual_files_changed=[{"path": "src/foo.py", "action": "modified"}],
        )
        excluded = _make_sample("excluded", triage_files=None)
        dataset = EvalDataset(samples=[qualifying, excluded])
        result = FilePredictionAccuracyMetric().compute(dataset, MetricConfig())
        assert result.details["sessions_with_file_predictions"] == 1


class TestFilePredictionAccuracyProperties:
    """Protocol attribute verification."""

    def test_properties(self) -> None:
        metric = FilePredictionAccuracyMetric()
        assert metric.name == "file_prediction_accuracy"
        assert metric.requires_ground_truth is False
        assert metric.requires_llm is False
        assert metric.higher_is_better is True
        assert metric.display_format == "percent"
        assert metric.display_name == "File prediction accuracy"
        assert isinstance(metric.description, str) and len(metric.description) > 0
        assert isinstance(metric.rationale, str) and len(metric.rationale) > 50
