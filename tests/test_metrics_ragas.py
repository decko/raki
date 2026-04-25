"""Tests for Ragas-backed retrieval quality metrics.

Adapter tests (to_ragas_rows, RagasRow) run without Ragas installed.
Metric tests mock ragas and run without it installed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raki.model import (
    EvalDataset,
    EvalSample,
    PhaseResult,
    SessionMeta,
)
from raki.model.ground_truth import GroundTruth
from raki.metrics.protocol import MetricConfig
from raki.docs.chunker import DocChunk
from raki.metrics.ragas.adapter import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    MAX_REFERENCE_CHARS,
    MAX_REFERENCE_CHUNKS,
    MAX_RESPONSE_CHARS,
    InstructorSilentZeroError,
    RagasRow,
    _extract_response_summary,
    is_instructor_silent_zero,
    select_relevant_chunks,
    to_ragas_rows,
    truncate_for_ragas,
)
from raki.metrics.ragas.llm_setup import JudgeLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_with_knowledge(
    session_id: str = "1",
    knowledge_context: str | None = "entry 1\n---\nentry 2",
    output: str = "The answer based on knowledge",
    ground_truth: GroundTruth | None = None,
) -> EvalSample:
    meta = SessionMeta(
        session_id=session_id,
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=2,
        rework_cycles=0,
    )
    triage = PhaseResult(
        name="triage",
        generation=1,
        status="completed",
        output="small",
        output_structured={"ticket_key": "101", "approach": "Add validation"},
    )
    implement = PhaseResult(
        name="implement",
        generation=1,
        status="completed",
        output=output,
        knowledge_context=knowledge_context,
    )
    return EvalSample(
        session=meta,
        phases=[triage, implement],
        findings=[],
        events=[],
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# Adapter tests — no ragas dependency needed
# ---------------------------------------------------------------------------


class TestRagasRow:
    def test_ragas_row_fields(self):
        row = RagasRow(
            session_id="s1",
            user_input="How to validate?",
            retrieved_contexts=["ctx1", "ctx2"],
            response="Use pydantic",
            reference="Use pydantic models",
        )
        assert row.session_id == "s1"
        assert row.user_input == "How to validate?"
        assert row.retrieved_contexts == ["ctx1", "ctx2"]
        assert row.response == "Use pydantic"
        assert row.reference == "Use pydantic models"

    def test_ragas_row_reference_none(self):
        row = RagasRow(
            session_id="s1",
            user_input="q",
            retrieved_contexts=["c"],
            response="r",
            reference=None,
        )
        assert row.reference is None


class TestToRagasRows:
    def test_extracts_contexts_with_ground_truth(self):
        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert rows[0].retrieved_contexts == ["entry 1", "entry 2"]
        assert rows[0].user_input == "How to validate?"
        # _extract_response_summary prefers triage approach over raw output
        assert "Add validation" in rows[0].response
        assert rows[0].reference == "Use pydantic"

    def test_uses_ticket_summary_when_no_ground_truth(self):
        sample = _make_sample_with_knowledge(ground_truth=None)
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert "Add validation" in rows[0].user_input
        assert rows[0].reference is None

    def test_skips_samples_without_knowledge_context(self):
        sample = _make_sample_with_knowledge(knowledge_context=None)
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 0

    def test_skips_samples_without_implement_phase(self):
        meta = SessionMeta(
            session_id="x",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
        )
        sample = EvalSample(
            session=meta,
            phases=[PhaseResult(name="triage", generation=1, status="completed", output="ok")],
            findings=[],
            events=[],
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 0

    def test_multiple_samples(self):
        gt1 = GroundTruth(question="Q1", reference_answer="A1", domains=[])
        gt2 = GroundTruth(question="Q2", reference_answer="A2", domains=[])
        sample1 = _make_sample_with_knowledge(session_id="s1", ground_truth=gt1)
        sample2 = _make_sample_with_knowledge(session_id="s2", ground_truth=gt2)
        no_knowledge = _make_sample_with_knowledge(session_id="s3", knowledge_context=None)
        rows = to_ragas_rows(EvalDataset(samples=[sample1, sample2, no_knowledge]))
        assert len(rows) == 2
        assert {row.session_id for row in rows} == {"s1", "s2"}

    def test_picks_latest_generation_implement_phase(self):
        meta = SessionMeta(
            session_id="gen-test",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=2,
            rework_cycles=1,
        )
        impl_gen1 = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="first attempt",
            knowledge_context="old context",
        )
        impl_gen2 = PhaseResult(
            name="implement",
            generation=2,
            status="completed",
            output="second attempt",
            knowledge_context="new ctx1\n---\nnew ctx2",
        )
        sample = EvalSample(
            session=meta,
            phases=[impl_gen1, impl_gen2],
            findings=[],
            events=[],
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert rows[0].response == "second attempt"
        assert rows[0].retrieved_contexts == ["new ctx1", "new ctx2"]

    def test_uses_session_phase_as_fallback(self):
        meta = SessionMeta(
            session_id="session-fallback",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
        )
        session_phase = PhaseResult(
            name="session",
            generation=1,
            status="completed",
            output="session output",
            knowledge_context="session knowledge",
        )
        sample = EvalSample(
            session=meta,
            phases=[session_phase],
            findings=[],
            events=[],
            ground_truth=GroundTruth(question="Q?", reference_answer="A", domains=[]),
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert rows[0].response == "session output"
        assert rows[0].retrieved_contexts == ["session knowledge"]


# ---------------------------------------------------------------------------
# Doc-chunk reference tests — doc chunks as reference for precision/recall
# ---------------------------------------------------------------------------


class TestToRagasRowsWithDocChunks:
    """When doc_chunks are provided, they should serve as reference for rows without ground truth."""

    def test_doc_chunks_used_as_reference_when_no_ground_truth(self):
        """Rows without ground_truth should get reference from doc_chunks."""
        sample = _make_sample_with_knowledge(ground_truth=None)
        doc_chunks = [
            DocChunk(
                text="Add authentication with JWT tokens",
                source_file="auth.md",
                domain="auth",
            ),
            DocChunk(
                text="Add validation with pydantic models",
                source_file="validation.md",
                domain="general",
            ),
        ]
        rows = to_ragas_rows(EvalDataset(samples=[sample]), doc_chunks=doc_chunks)
        assert len(rows) == 1
        assert rows[0].reference is not None
        assert "authentication with JWT tokens" in rows[0].reference
        assert "validation with pydantic models" in rows[0].reference

    def test_ground_truth_takes_precedence_over_doc_chunks(self):
        """When ground_truth has reference_answer, it should be used instead of doc_chunks."""
        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        doc_chunks = [
            DocChunk(text="Some doc text", source_file="doc.md", domain="general"),
        ]
        rows = to_ragas_rows(EvalDataset(samples=[sample]), doc_chunks=doc_chunks)
        assert len(rows) == 1
        assert rows[0].reference == "Use pydantic"

    def test_no_doc_chunks_no_ground_truth_reference_is_none(self):
        """Without doc_chunks or ground_truth, reference should remain None."""
        sample = _make_sample_with_knowledge(ground_truth=None)
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert rows[0].reference is None

    def test_empty_doc_chunks_no_reference(self):
        """Empty doc_chunks list should not set reference."""
        sample = _make_sample_with_knowledge(ground_truth=None)
        rows = to_ragas_rows(EvalDataset(samples=[sample]), doc_chunks=[])
        assert len(rows) == 1
        assert rows[0].reference is None


class TestPrecisionWithDocChunks:
    """Context precision should compute real scores when doc_chunks provide reference."""

    def test_computes_with_doc_chunks_no_ground_truth(self, monkeypatch: pytest.MonkeyPatch):
        """Precision should score samples using doc_chunks as reference."""
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        mock_result = MagicMock()
        mock_result.value = 0.80
        mock_result.reason = "Precise with doc chunks"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])

        doc_chunks = [
            DocChunk(
                text="Add validation for reference doc content",
                source_file="ref.md",
                domain="general",
            ),
        ]
        config = MetricConfig(doc_chunks=doc_chunks)

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.80)
        assert result.details["samples_scored"] == 1
        assert "skipped" not in result.details

    def test_still_skips_without_doc_chunks_or_ground_truth(self):
        """Without doc_chunks or ground_truth, precision should still return N/A."""
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None
        assert result.details.get("skipped") == "no ground truth"


class TestRecallWithDocChunks:
    """Context recall should compute real scores when doc_chunks provide reference."""

    def test_computes_with_doc_chunks_no_ground_truth(self, monkeypatch: pytest.MonkeyPatch):
        """Recall should score samples using doc_chunks as reference."""
        from raki.metrics.ragas.recall import ContextRecallMetric

        mock_result = MagicMock()
        mock_result.value = 0.75
        mock_result.reason = "Good recall with doc chunks"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])

        doc_chunks = [
            DocChunk(
                text="Add validation for reference doc content",
                source_file="ref.md",
                domain="general",
            ),
        ]
        config = MetricConfig(doc_chunks=doc_chunks)

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.75)
        assert result.details["samples_scored"] == 1
        assert "skipped" not in result.details

    def test_still_skips_without_doc_chunks_or_ground_truth(self):
        """Without doc_chunks or ground_truth, recall should still return N/A."""
        from raki.metrics.ragas.recall import ContextRecallMetric

        metric = ContextRecallMetric()
        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None
        assert result.details.get("skipped") == "no ground truth"


# ---------------------------------------------------------------------------
# JudgeLogger tests — no ragas dependency needed
# ---------------------------------------------------------------------------


class TestJudgeLogger:
    def test_logs_to_jsonl(self, tmp_path: Path):
        log_path = tmp_path / "subdir" / "judge_log.jsonl"
        judge_logger = JudgeLogger(log_path, project_root=tmp_path)
        judge_logger.log("context_precision", "How to validate?", 0.85, "Good precision")
        judge_logger.log("context_recall", "How to test?", 0.72, None)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["metric"] == "context_precision"
        assert entry1["score"] == 0.85
        assert entry1["reason"] == "Good precision"

        entry2 = json.loads(lines[1])
        assert entry2["metric"] == "context_recall"
        assert entry2["score"] == 0.72
        assert entry2["reason"] is None

    def test_truncates_long_user_input(self, tmp_path: Path):
        log_path = tmp_path / "judge_log.jsonl"
        judge_logger = JudgeLogger(log_path, project_root=tmp_path)
        long_input = "x" * 500
        judge_logger.log("test_metric", long_input, 0.5, None)

        entry = json.loads(log_path.read_text().strip())
        assert len(entry["user_input"]) == 200

    def test_rejects_path_outside_project_root(self, tmp_path: Path):
        """_validate_judge_log_path should use project_root, not Path.cwd()."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        outside_path = tmp_path / "outside" / "judge.jsonl"
        with pytest.raises(ValueError, match="escapes project root"):
            JudgeLogger(outside_path, project_root=project_dir)

    def test_accepts_path_inside_project_root(self, tmp_path: Path):
        """Log path under project_root should be accepted without needing monkeypatch.chdir."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        log_path = project_dir / "logs" / "judge.jsonl"
        judge_logger = JudgeLogger(log_path, project_root=project_dir)
        assert judge_logger.log_path == log_path.resolve()


# ---------------------------------------------------------------------------
# Metric tests — mock ragas so tests run without it installed
# ---------------------------------------------------------------------------


def _install_ragas_mock(
    monkeypatch: pytest.MonkeyPatch,
    mock_metric_class: MagicMock,
    ragas_class_name: str,
) -> None:
    """Insert a fake ``ragas.metrics.collections`` module into sys.modules.

    This lets ``from ragas.metrics.collections import <ragas_class_name>``
    succeed inside ``compute()`` without ragas installed.
    """
    import sys

    fake_collections = MagicMock()
    setattr(fake_collections, ragas_class_name, mock_metric_class)
    monkeypatch.setitem(sys.modules, "ragas", MagicMock())
    monkeypatch.setitem(sys.modules, "ragas.metrics", MagicMock())
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", fake_collections)


class TestContextPrecisionMetric:
    def test_skips_without_ground_truth(self):
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None
        assert result.details.get("skipped") == "no ground truth"

    def test_computes_with_mocked_ragas(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        monkeypatch.chdir(tmp_path)

        mock_result = MagicMock()
        mock_result.value = 0.85
        mock_result.reason = "Good precision"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.name == "context_precision"
        assert result.score == pytest.approx(0.85)
        assert result.details["samples_scored"] == 1
        assert "1" in result.sample_scores

        # Verify judge log was written
        log_lines = log_path.read_text().strip().split("\n")
        assert len(log_lines) == 1
        log_entry = json.loads(log_lines[0])
        assert log_entry["metric"] == "context_precision"
        assert log_entry["score"] == 0.85

    def test_handles_float_return_from_ascore(self, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=0.75)

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.75)

    def test_ascore_called_with_correct_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        """Verify ascore() receives user_input, retrieved_contexts, reference as kwargs."""
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        mock_result = MagicMock()
        mock_result.value = 0.90
        mock_result.reason = None

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            metric.compute(dataset, config)

        mock_metric_instance.ascore.assert_called_once()
        call_kwargs = mock_metric_instance.ascore.call_args.kwargs
        assert call_kwargs["user_input"] == "How to validate?"
        assert call_kwargs["retrieved_contexts"] == ["entry 1", "entry 2"]
        assert call_kwargs["reference"] == "Use pydantic"

    def test_logs_errors_on_ascore_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=RuntimeError("LLM down"))

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.score == 0.0
        assert result.details["samples_scored"] == 0

        log_entry = json.loads(log_path.read_text().strip())
        assert log_entry["score"] == -1.0
        assert "LLM down" in log_entry["reason"]


class TestContextRecallMetric:
    def test_skips_without_ground_truth(self):
        from raki.metrics.ragas.recall import ContextRecallMetric

        metric = ContextRecallMetric()
        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None
        assert result.details.get("skipped") == "no ground truth"

    def test_computes_with_mocked_ragas(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.recall import ContextRecallMetric

        monkeypatch.chdir(tmp_path)

        mock_result = MagicMock()
        mock_result.value = 0.92
        mock_result.reason = "High recall"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            result = metric.compute(dataset, config)

        assert result.name == "context_recall"
        assert result.score == pytest.approx(0.92)
        assert result.details["samples_scored"] == 1

    def test_handles_float_return_from_ascore(self, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.recall import ContextRecallMetric

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=0.88)

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.88)

    def test_ascore_called_with_correct_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        """Verify ascore() receives user_input, retrieved_contexts, reference as kwargs."""
        from raki.metrics.ragas.recall import ContextRecallMetric

        mock_result = MagicMock()
        mock_result.value = 0.91
        mock_result.reason = None

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            metric.compute(dataset, config)

        mock_metric_instance.ascore.assert_called_once()
        call_kwargs = mock_metric_instance.ascore.call_args.kwargs
        assert call_kwargs["user_input"] == "How to validate?"
        assert call_kwargs["retrieved_contexts"] == ["entry 1", "entry 2"]
        assert call_kwargs["reference"] == "Use pydantic"


# ---------------------------------------------------------------------------
# Async safety tests — verify _run_async works in different contexts
# ---------------------------------------------------------------------------


class TestRunAsync:
    def test_run_async_outside_event_loop(self):
        from raki.metrics.ragas.async_utils import run_async

        async def simple_coro():
            return 42

        assert run_async(simple_coro()) == 42

    def test_run_async_inside_event_loop(self):
        """Verify run_async works even when called from within an event loop."""
        import asyncio

        from raki.metrics.ragas.async_utils import run_async

        async def inner():
            return 99

        async def outer():
            return run_async(inner())

        result = asyncio.run(outer())
        assert result == 99


# ---------------------------------------------------------------------------
# MetricConfig judge_log_path test
# ---------------------------------------------------------------------------


class TestMetricConfigJudgeLogPath:
    def test_judge_log_path_default_none(self):
        config = MetricConfig()
        assert config.judge_log_path is None

    def test_judge_log_path_set(self, tmp_path: Path):
        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)
        assert config.judge_log_path == log_path


# ---------------------------------------------------------------------------
# No legacy mock helpers -- all 4 metrics now use the collections API
# via _install_ragas_mock() above.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Faithfulness metric tests — mock ragas so tests run without it installed
# ---------------------------------------------------------------------------


class TestFaithfulnessMetric:
    def test_returns_none_score_when_no_retrieval_context(self):
        """Faithfulness should return score=None when no context is available."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric()
        sample = _make_sample_with_knowledge(knowledge_context=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None
        assert result.details.get("skipped") == "no retrieval context"

    def test_skips_without_samples(self):
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric()
        sample = _make_sample_with_knowledge(knowledge_context=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None

    def test_experimental_flag(self):
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric()
        assert metric.experimental is True

    def test_protocol_properties(self):
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric()
        assert metric.name == "faithfulness"
        assert metric.requires_ground_truth is False
        assert metric.requires_llm is True
        assert metric.higher_is_better is True
        assert metric.display_format == "score"
        assert metric.display_name == "Faithfulness"

    def test_computes_with_mocked_ragas(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Faithfulness uses collections API ascore() with keyword args."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        monkeypatch.chdir(tmp_path)

        mock_result = MagicMock()
        mock_result.value = 0.9
        mock_result.reason = "Faithful response"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.name == "faithfulness"
        assert result.score == pytest.approx(0.9)
        assert result.details["samples_scored"] == 1
        assert result.details["experimental"] is True
        assert "NL answers" in result.details["caveat"]
        assert "1" in result.sample_scores

        log_lines = log_path.read_text().strip().split("\n")
        assert len(log_lines) == 1
        log_entry = json.loads(log_lines[0])
        assert log_entry["metric"] == "faithfulness"
        assert log_entry["score"] == 0.9

    def test_ascore_called_with_correct_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        """Verify ascore() receives user_input, response, retrieved_contexts as kwargs."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        mock_result = MagicMock()
        mock_result.value = 0.85
        mock_result.reason = None

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            metric.compute(dataset, config)

        mock_metric_instance.ascore.assert_called_once()
        call_kwargs = mock_metric_instance.ascore.call_args.kwargs
        assert call_kwargs["user_input"] == "How to validate?"
        # _extract_response_summary picks up triage approach from the sample
        assert "Add validation" in call_kwargs["response"]
        assert call_kwargs["retrieved_contexts"] == ["entry 1", "entry 2"]

    def test_handles_float_return_from_ascore(self, monkeypatch: pytest.MonkeyPatch):
        """Faithfulness handles both MetricResult and plain float from ascore()."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=0.77)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.77)

    def test_does_not_require_ground_truth(self, monkeypatch: pytest.MonkeyPatch):
        """Faithfulness scores all samples, not just those with ground truth."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        mock_result = MagicMock()
        mock_result.value = 0.8
        mock_result.reason = None

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge(ground_truth=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.8)
        assert result.details["samples_scored"] == 1

    def test_handles_ascore_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score == 0.0
        assert result.details["samples_scored"] == 0

        log_entry = json.loads(log_path.read_text().strip())
        assert log_entry["score"] == -1.0
        assert "LLM unavailable" in log_entry["reason"]


# ---------------------------------------------------------------------------
# AnswerRelevancy metric tests — mock ragas so tests run without it installed
# ---------------------------------------------------------------------------


class TestAnswerRelevancyMetric:
    def test_returns_none_score_when_no_retrieval_context(self):
        """AnswerRelevancy should return score=None when no context is available."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        sample = _make_sample_with_knowledge(knowledge_context=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None
        assert result.details.get("skipped") == "no retrieval context"

    def test_skips_without_samples(self):
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        sample = _make_sample_with_knowledge(knowledge_context=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score is None

    def test_experimental_flag(self):
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        assert metric.experimental is True

    def test_protocol_properties(self):
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        assert metric.name == "answer_relevancy"
        assert metric.requires_ground_truth is False
        assert metric.requires_llm is True
        assert metric.higher_is_better is True
        assert metric.display_format == "score"
        assert metric.display_name == "Answer relevancy"

    def test_computes_with_mocked_ragas(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """AnswerRelevancy uses collections API ascore() with keyword args."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        monkeypatch.chdir(tmp_path)

        mock_result = MagicMock()
        mock_result.value = 0.78
        mock_result.reason = "Relevant answer"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            result = metric.compute(dataset, config)

        assert result.name == "answer_relevancy"
        assert result.score == pytest.approx(0.78)
        assert result.details["samples_scored"] == 1
        assert result.details["experimental"] is True
        assert "NL answers" in result.details["caveat"]
        assert "1" in result.sample_scores

        log_lines = log_path.read_text().strip().split("\n")
        assert len(log_lines) == 1
        log_entry = json.loads(log_lines[0])
        assert log_entry["metric"] == "answer_relevancy"
        assert log_entry["score"] == 0.78

    def test_ascore_called_with_correct_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        """Verify ascore() receives user_input and response as kwargs."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        mock_result = MagicMock()
        mock_result.value = 0.82
        mock_result.reason = None

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            metric.compute(dataset, config)

        mock_metric_instance.ascore.assert_called_once()
        call_kwargs = mock_metric_instance.ascore.call_args.kwargs
        assert call_kwargs["user_input"] == "How to validate?"
        # _extract_response_summary picks up triage approach from the sample
        assert "Add validation" in call_kwargs["response"]
        assert "retrieved_contexts" not in call_kwargs

    def test_handles_float_return_from_ascore(self, monkeypatch: pytest.MonkeyPatch):
        """AnswerRelevancy handles both MetricResult and plain float from ascore()."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=0.65)

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.65)

    def test_handles_ascore_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=RuntimeError("Embeddings failed"))

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])

        log_path = tmp_path / "judge.jsonl"
        config = MetricConfig(judge_log_path=log_path)

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            result = metric.compute(dataset, config)

        assert result.score == 0.0
        assert result.details["samples_scored"] == 0

        log_entry = json.loads(log_path.read_text().strip())
        assert log_entry["score"] == -1.0


# ---------------------------------------------------------------------------
# LLM setup tests — verify create_ragas_llm and create_ragas_embeddings
# ---------------------------------------------------------------------------


class TestCreateRagasLlm:
    def test_passes_temperature_to_llm_factory(self, monkeypatch: pytest.MonkeyPatch):
        """config.temperature must be forwarded to llm_factory()."""
        import sys

        mock_llm_factory = MagicMock(return_value=MagicMock())
        mock_llms = MagicMock()
        mock_llms.llm_factory = mock_llm_factory

        mock_anthropic = MagicMock()
        mock_client_instance = MagicMock()
        mock_anthropic.AsyncAnthropicVertex = MagicMock(return_value=mock_client_instance)

        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
        monkeypatch.setitem(sys.modules, "ragas", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.llms", mock_llms)

        from raki.metrics.ragas.llm_setup import create_ragas_llm

        config = MetricConfig(temperature=0.7)
        create_ragas_llm(config)

        mock_llm_factory.assert_called_once()
        call_kwargs = mock_llm_factory.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_passes_zero_temperature_by_default(self, monkeypatch: pytest.MonkeyPatch):
        """Default temperature=0.0 should also be forwarded."""
        import sys

        mock_llm_factory = MagicMock(return_value=MagicMock())
        mock_llms = MagicMock()
        mock_llms.llm_factory = mock_llm_factory

        mock_anthropic = MagicMock()
        mock_client_instance = MagicMock()
        mock_anthropic.AsyncAnthropicVertex = MagicMock(return_value=mock_client_instance)

        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
        monkeypatch.setitem(sys.modules, "ragas", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.llms", mock_llms)

        from raki.metrics.ragas.llm_setup import create_ragas_llm

        config = MetricConfig()  # default temperature=0.0
        create_ragas_llm(config)

        call_kwargs = mock_llm_factory.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0


class TestLLMProviderDispatch:
    """Tests for provider dispatch in create_ragas_llm()."""

    def test_vertex_anthropic_uses_async_anthropic_vertex(self, monkeypatch: pytest.MonkeyPatch):
        """vertex-anthropic provider should instantiate AsyncAnthropicVertex."""
        import sys

        mock_llm_factory = MagicMock(return_value=MagicMock())
        mock_llms = MagicMock()
        mock_llms.llm_factory = mock_llm_factory

        mock_anthropic = MagicMock()
        mock_vertex_instance = MagicMock()
        mock_anthropic.AsyncAnthropicVertex = MagicMock(return_value=mock_vertex_instance)

        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
        monkeypatch.setitem(sys.modules, "ragas", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.llms", mock_llms)

        from raki.metrics.ragas.llm_setup import create_ragas_llm

        config = MetricConfig(llm_provider="vertex-anthropic")
        create_ragas_llm(config)

        mock_anthropic.AsyncAnthropicVertex.assert_called_once()
        mock_llm_factory.assert_called_once()
        call_kwargs = mock_llm_factory.call_args
        assert call_kwargs.kwargs["client"] is mock_vertex_instance

    def test_anthropic_uses_async_anthropic(self, monkeypatch: pytest.MonkeyPatch):
        """anthropic provider should instantiate AsyncAnthropic (direct API)."""
        import sys

        mock_llm_factory = MagicMock(return_value=MagicMock())
        mock_llms = MagicMock()
        mock_llms.llm_factory = mock_llm_factory

        mock_anthropic = MagicMock()
        mock_direct_instance = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_direct_instance)

        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
        monkeypatch.setitem(sys.modules, "ragas", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.llms", mock_llms)

        from raki.metrics.ragas.llm_setup import create_ragas_llm

        config = MetricConfig(llm_provider="anthropic")
        create_ragas_llm(config)

        mock_anthropic.AsyncAnthropic.assert_called_once()
        mock_llm_factory.assert_called_once()
        call_kwargs = mock_llm_factory.call_args
        assert call_kwargs.kwargs["client"] is mock_direct_instance

    def test_unknown_provider_rejected_by_pydantic(self):
        """Pydantic Literal type rejects unknown providers at config construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="literal_error"):
            MetricConfig(llm_provider="openai")  # type: ignore[arg-type]

    def test_unknown_provider_error_lists_valid_options(self):
        """Pydantic error message should mention supported providers."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="vertex-anthropic") as exc_info:
            MetricConfig(llm_provider="bedrock")  # type: ignore[arg-type]
        assert "anthropic" in str(exc_info.value)

    def test_create_ragas_llm_raises_on_invalid_provider(self):
        """Defense-in-depth: create_ragas_llm raises ValueError for unknown providers."""
        from raki.metrics.ragas.llm_setup import create_ragas_llm

        # Bypass Pydantic validation to test the runtime guard
        config = MetricConfig()
        object.__setattr__(config, "llm_provider", "unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_ragas_llm(config)

    def test_default_provider_is_vertex_anthropic(self):
        """MetricConfig default llm_provider should be vertex-anthropic."""
        config = MetricConfig()
        assert config.llm_provider == "vertex-anthropic"


class TestCreateRagasEmbeddings:
    """Verify GoogleEmbeddings is constructed with use_vertex=True and the pre-configured client."""

    def _setup_embeddings_mocks(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Set up common mocks for embeddings tests, returning mock objects for assertions."""
        import importlib
        import sys

        mock_genai_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_genai_client

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        mock_google_embeddings_instance = MagicMock()
        mock_google_embeddings_class = MagicMock(return_value=mock_google_embeddings_instance)
        mock_google_provider = MagicMock()
        mock_google_provider.GoogleEmbeddings = mock_google_embeddings_class

        monkeypatch.setitem(sys.modules, "google", mock_google)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
        monkeypatch.setitem(sys.modules, "ragas", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.embeddings", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.embeddings.google_provider", mock_google_provider)

        import raki.metrics.ragas.llm_setup

        importlib.reload(raki.metrics.ragas.llm_setup)

        return {
            "genai": mock_genai,
            "genai_client": mock_genai_client,
            "embeddings_class": mock_google_embeddings_class,
            "embeddings_instance": mock_google_embeddings_instance,
            "create_fn": raki.metrics.ragas.llm_setup.create_ragas_embeddings,
        }

    def test_embeddings_use_vertex_true(self, monkeypatch: pytest.MonkeyPatch):
        """GoogleEmbeddings must be constructed with use_vertex=True, client, and model."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vertex-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-east1")

        mocks = self._setup_embeddings_mocks(monkeypatch)
        result = mocks["create_fn"]()

        mocks["embeddings_class"].assert_called_once_with(
            client=mocks["genai_client"],
            model="text-embedding-005",
            use_vertex=True,
        )
        assert result is mocks["embeddings_instance"]

    def test_embeddings_passes_client_through(self, monkeypatch: pytest.MonkeyPatch):
        """The pre-configured genai.Client must not be discarded; it must reach GoogleEmbeddings."""
        monkeypatch.setenv("VERTEXAI_PROJECT", "client-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "europe-west4")

        mocks = self._setup_embeddings_mocks(monkeypatch)
        mocks["create_fn"]()

        mocks["genai"].Client.assert_called_once_with(
            vertexai=True, project="client-project", location="europe-west4"
        )

        call_kwargs = mocks["embeddings_class"].call_args.kwargs
        assert call_kwargs["client"] is mocks["genai_client"]
        assert call_kwargs["use_vertex"] is True

    def test_passes_vertex_project_to_genai_client(self, monkeypatch: pytest.MonkeyPatch):
        """Verify genai.Client is created with vertexai=True and project from env."""
        monkeypatch.setenv("VERTEXAI_PROJECT", "my-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "europe-west1")

        mocks = self._setup_embeddings_mocks(monkeypatch)
        mocks["create_fn"]()

        mocks["genai"].Client.assert_called_once_with(
            vertexai=True, project="my-project", location="europe-west1"
        )

    def test_google_cloud_project_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        """When both GOOGLE_CLOUD_PROJECT and VERTEXAI_PROJECT are set, GOOGLE_CLOUD_PROJECT wins."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "primary-project")
        monkeypatch.setenv("VERTEXAI_PROJECT", "fallback-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

        mocks = self._setup_embeddings_mocks(monkeypatch)
        mocks["create_fn"]()

        mocks["genai"].Client.assert_called_once_with(
            vertexai=True, project="primary-project", location="us-central1"
        )

    def test_vertexai_project_fallback(self, monkeypatch: pytest.MonkeyPatch):
        """VERTEXAI_PROJECT is used when GOOGLE_CLOUD_PROJECT is not set."""
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.setenv("VERTEXAI_PROJECT", "fallback-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

        mocks = self._setup_embeddings_mocks(monkeypatch)
        mocks["create_fn"]()

        mocks["genai"].Client.assert_called_once_with(
            vertexai=True, project="fallback-project", location="us-central1"
        )

    def test_missing_project_raises_valueerror(self, monkeypatch: pytest.MonkeyPatch):
        """When neither GOOGLE_CLOUD_PROJECT nor VERTEXAI_PROJECT is set, raise ValueError."""
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

        mocks = self._setup_embeddings_mocks(monkeypatch)

        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT"):
            mocks["create_fn"]()


# ---------------------------------------------------------------------------
# Google provider tests — verify create_ragas_llm dispatches to Google
# ---------------------------------------------------------------------------


class TestGoogleProvider:
    def test_google_provider_creates_llm(self, monkeypatch):
        """Google provider dispatches to llm_factory with provider='google'."""
        import sys

        mock_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        mock_llm = MagicMock()
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

        result = llm_setup.create_ragas_llm(config)

        mock_genai.Client.assert_called_once_with(
            vertexai=True, project="test-project", location="us-central1"
        )
        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args
        assert call_kwargs[0][0] == "gemini-2.5-pro"
        assert call_kwargs[1]["provider"] == "google"
        assert call_kwargs[1]["client"] == mock_client
        assert result is mock_llm

    def test_google_in_supported_providers(self):
        """google is listed in SUPPORTED_PROVIDERS."""
        from raki.metrics.ragas.llm_setup import SUPPORTED_PROVIDERS

        assert "google" in SUPPORTED_PROVIDERS

    def test_google_provider_missing_project_raises(self, monkeypatch):
        """When neither GOOGLE_CLOUD_PROJECT nor VERTEXAI_PROJECT is set, raise ValueError."""
        import sys

        mock_genai = MagicMock()
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        mock_factory = MagicMock(return_value=MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT"):
            llm_setup.create_ragas_llm(config)

    def test_google_provider_vertexai_project_fallback(self, monkeypatch):
        """VERTEXAI_PROJECT is used when GOOGLE_CLOUD_PROJECT is not set."""
        import sys

        mock_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        mock_llm = MagicMock()
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.setenv("VERTEXAI_PROJECT", "fallback-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

        llm_setup.create_ragas_llm(config)

        mock_genai.Client.assert_called_once_with(
            vertexai=True, project="fallback-project", location="us-central1"
        )

    def test_google_provider_forwards_temperature(self, monkeypatch):
        """Temperature from MetricConfig is passed to llm_factory."""
        import sys

        mock_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        mock_llm = MagicMock()
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro", temperature=0.5)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

        llm_setup.create_ragas_llm(config)

        call_kwargs = mock_factory.call_args
        assert call_kwargs[1]["temperature"] == 0.5

    def test_google_provider_default_location(self, monkeypatch):
        """When VERTEXAI_LOCATION is not set, 'us-central1' is used as default."""
        import sys

        mock_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        mock_llm = MagicMock()
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)

        llm_setup.create_ragas_llm(config)

        mock_genai.Client.assert_called_once_with(
            vertexai=True, project="test-project", location="us-central1"
        )


# ---------------------------------------------------------------------------
# LiteLLM provider tests — verify create_ragas_llm dispatches to LiteLLM
# ---------------------------------------------------------------------------


class TestLiteLLMProvider:
    """Tests for the 'litellm' provider branch in create_ragas_llm()."""

    def _setup_litellm_mocks(self, monkeypatch):
        """Set up common mocks for litellm provider tests."""
        import sys
        from importlib import reload

        mock_litellm = MagicMock()
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        mock_llm = MagicMock()
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        return {
            "litellm": mock_litellm,
            "llm": mock_llm,
            "factory": mock_factory,
            "module": llm_setup,
        }

    def test_litellm_provider_creates_llm(self, monkeypatch):
        """litellm provider dispatches to llm_factory with provider='litellm'."""
        mocks = self._setup_litellm_mocks(monkeypatch)

        config = MetricConfig(llm_provider="litellm", llm_model="gpt-4o")
        result = mocks["module"].create_ragas_llm(config)

        mocks["factory"].assert_called_once()
        call_kwargs = mocks["factory"].call_args
        assert call_kwargs[0][0] == "gpt-4o"
        assert call_kwargs[1]["provider"] == "litellm"
        assert result is mocks["llm"]

    def test_litellm_provider_passes_litellm_module_as_client(self, monkeypatch):
        """The litellm module itself must be passed as the client."""
        mocks = self._setup_litellm_mocks(monkeypatch)

        config = MetricConfig(llm_provider="litellm", llm_model="gpt-4o")
        mocks["module"].create_ragas_llm(config)

        call_kwargs = mocks["factory"].call_args
        assert call_kwargs[1]["client"] is mocks["litellm"]

    def test_litellm_provider_forwards_temperature(self, monkeypatch):
        """Temperature from MetricConfig is passed to llm_factory."""
        mocks = self._setup_litellm_mocks(monkeypatch)

        config = MetricConfig(llm_provider="litellm", llm_model="gpt-4o", temperature=0.3)
        mocks["module"].create_ragas_llm(config)

        call_kwargs = mocks["factory"].call_args
        assert call_kwargs[1]["temperature"] == 0.3

    def test_litellm_provider_removes_top_p(self, monkeypatch):
        """top_p must be removed from model_args to avoid backend rejection."""
        mocks = self._setup_litellm_mocks(monkeypatch)

        # Give the mock llm a real model_args dict so pop is testable
        mock_llm = mocks["llm"]
        mock_llm.model_args = {"top_p": 0.9, "temperature": 0.0}
        mocks["factory"].return_value = mock_llm

        config = MetricConfig(llm_provider="litellm", llm_model="gpt-4o")
        mocks["module"].create_ragas_llm(config)

        assert "top_p" not in mock_llm.model_args

    def test_litellm_in_supported_providers(self):
        """'litellm' must appear in SUPPORTED_PROVIDERS."""
        from raki.metrics.ragas.llm_setup import SUPPORTED_PROVIDERS

        assert "litellm" in SUPPORTED_PROVIDERS

    def test_litellm_provider_no_token_accumulator_skips_patch(self, monkeypatch):
        """When token_accumulator is None, acompletion must not be patched."""
        import sys
        from importlib import reload

        mock_litellm = MagicMock()
        original_acompletion = mock_litellm.acompletion
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)
        monkeypatch.setitem(
            sys.modules, "ragas.llms", MagicMock(llm_factory=MagicMock(return_value=MagicMock()))
        )

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="litellm", llm_model="gpt-4o", token_accumulator=None)
        llm_setup.create_ragas_llm(config)

        # acompletion must not have been replaced
        assert mock_litellm.acompletion is original_acompletion


class TestPatchLiteLLMForTokenTracking:
    """Tests for patch_litellm_for_token_tracking()."""

    def _make_usage(self, prompt_tokens: int, completion_tokens: int):
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        return usage

    def test_tracks_input_and_output_tokens(self):
        """Wrapping acompletion should increment input_tokens and output_tokens."""
        import asyncio

        from raki.metrics.protocol import TokenAccumulator
        from raki.metrics.ragas.llm_setup import patch_litellm_for_token_tracking

        accumulator = TokenAccumulator()
        mock_response = MagicMock()
        mock_response.usage = self._make_usage(prompt_tokens=10, completion_tokens=5)

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        patch_litellm_for_token_tracking(mock_litellm, accumulator)
        asyncio.run(mock_litellm.acompletion(model="gpt-4o", messages=[]))

        assert accumulator.input_tokens == 10
        assert accumulator.output_tokens == 5
        assert accumulator.calls == 1

    def test_accumulates_across_multiple_calls(self):
        """Accumulator totals should grow with each acompletion call."""
        import asyncio

        from raki.metrics.protocol import TokenAccumulator
        from raki.metrics.ragas.llm_setup import patch_litellm_for_token_tracking

        accumulator = TokenAccumulator()
        mock_litellm = MagicMock()

        call_count = 0

        async def fake_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.usage = self._make_usage(prompt_tokens=4, completion_tokens=2)
            return resp

        mock_litellm.acompletion = fake_acompletion

        patch_litellm_for_token_tracking(mock_litellm, accumulator)
        asyncio.run(mock_litellm.acompletion())
        asyncio.run(mock_litellm.acompletion())
        asyncio.run(mock_litellm.acompletion())

        assert accumulator.input_tokens == 12
        assert accumulator.output_tokens == 6
        assert accumulator.calls == 3

    def test_missing_usage_does_not_crash(self):
        """When the response has no usage attribute, accumulator.calls still increments."""
        import asyncio

        from raki.metrics.protocol import TokenAccumulator
        from raki.metrics.ragas.llm_setup import patch_litellm_for_token_tracking

        accumulator = TokenAccumulator()
        mock_response = MagicMock(spec=[])  # no attributes at all
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        patch_litellm_for_token_tracking(mock_litellm, accumulator)
        asyncio.run(mock_litellm.acompletion())

        assert accumulator.input_tokens == 0
        assert accumulator.output_tokens == 0
        assert accumulator.calls == 1

    def test_none_usage_does_not_crash(self):
        """When response.usage is None, accumulator.calls still increments."""
        import asyncio

        from raki.metrics.protocol import TokenAccumulator
        from raki.metrics.ragas.llm_setup import patch_litellm_for_token_tracking

        accumulator = TokenAccumulator()
        mock_response = MagicMock()
        mock_response.usage = None
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        patch_litellm_for_token_tracking(mock_litellm, accumulator)
        asyncio.run(mock_litellm.acompletion())

        assert accumulator.input_tokens == 0
        assert accumulator.output_tokens == 0
        assert accumulator.calls == 1

    def test_original_response_returned_unmodified(self):
        """The response object must be returned intact."""
        import asyncio

        from raki.metrics.protocol import TokenAccumulator
        from raki.metrics.ragas.llm_setup import patch_litellm_for_token_tracking

        sentinel = object()
        mock_response = MagicMock()
        mock_response.usage = self._make_usage(1, 1)
        mock_response._sentinel = sentinel

        accumulator = TokenAccumulator()
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        patch_litellm_for_token_tracking(mock_litellm, accumulator)
        result = asyncio.run(mock_litellm.acompletion())

        assert result is mock_response

    def test_litellm_provider_with_accumulator_patches_acompletion(self, monkeypatch):
        """When token_accumulator is set, acompletion is replaced on the module."""
        import sys
        from importlib import reload

        from raki.metrics.protocol import TokenAccumulator

        accumulator = TokenAccumulator()
        mock_litellm = MagicMock()
        original_acompletion = mock_litellm.acompletion
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)
        monkeypatch.setitem(
            sys.modules, "ragas.llms", MagicMock(llm_factory=MagicMock(return_value=MagicMock()))
        )

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(
            llm_provider="litellm", llm_model="gpt-4o", token_accumulator=accumulator
        )
        llm_setup.create_ragas_llm(config)

        # acompletion must have been replaced by the wrapper
        assert mock_litellm.acompletion is not original_acompletion


# ---------------------------------------------------------------------------
# instructor#1658 silent-zero detection tests
# ---------------------------------------------------------------------------


class TestIsInstructorSilentZero:
    """Tests for the is_instructor_silent_zero() detection function."""

    def test_returns_true_for_google_zero_no_reason(self):
        """Google provider + zero value + no reason → silent-zero detected."""
        result = MagicMock()
        result.value = 0.0
        result.reason = None
        assert is_instructor_silent_zero(result, "google") is True

    def test_returns_true_for_google_zero_empty_reason(self):
        """Empty-string reason is also considered 'no reason'."""
        result = MagicMock()
        result.value = 0.0
        result.reason = ""
        assert is_instructor_silent_zero(result, "google") is True

    def test_returns_false_for_google_zero_with_reason(self):
        """When reason is present, 0.0 is a legitimate score, not a silent failure."""
        result = MagicMock()
        result.value = 0.0
        result.reason = "No relevant context found for the question"
        assert is_instructor_silent_zero(result, "google") is False

    def test_returns_false_for_google_nonzero_no_reason(self):
        """Non-zero value with no reason is not the silent-zero bug."""
        result = MagicMock()
        result.value = 0.85
        result.reason = None
        assert is_instructor_silent_zero(result, "google") is False

    def test_returns_false_for_anthropic_zero_no_reason(self):
        """Silent-zero detection applies only to the google provider."""
        result = MagicMock()
        result.value = 0.0
        result.reason = None
        assert is_instructor_silent_zero(result, "anthropic") is False

    def test_returns_false_for_vertex_anthropic_zero_no_reason(self):
        """vertex-anthropic provider is not affected by instructor#1658."""
        result = MagicMock()
        result.value = 0.0
        result.reason = None
        assert is_instructor_silent_zero(result, "vertex-anthropic") is False

    def test_returns_false_for_plain_float_result(self):
        """A plain float return from ascore() bypasses structured-output parsing entirely."""
        assert is_instructor_silent_zero(0.0, "google") is False

    def test_instructor_silent_zero_error_is_runtime_error(self):
        """InstructorSilentZeroError must subclass RuntimeError for consistent handling."""
        exc = InstructorSilentZeroError("test message")
        assert isinstance(exc, RuntimeError)

    def test_instructor_silent_zero_error_message(self):
        """Error message should mention instructor#1658 for traceability."""
        msg = "instructor#1658: silent zero"
        exc = InstructorSilentZeroError(msg)
        assert "instructor#1658" in str(exc)


class TestSilentZeroHandlingFaithfulness:
    """Faithfulness metric skips sessions with instructor#1658 silent-zero scores."""

    def test_silent_zero_skipped_not_averaged(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with no reason, session is skipped, not averaged."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = None  # instructor#1658 silent zero

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "instructor#1658" in result.details.get("skipped", "")
        assert result.details.get("silent_zero_sessions") == 1

    def test_legitimate_zero_from_google_with_reason_is_kept(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with a reason, it is a real score and must be included."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = "No faithful statements found"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.0)
        assert result.details["samples_scored"] == 1

    def test_non_google_zero_no_reason_is_kept(self, monkeypatch: pytest.MonkeyPatch):
        """Silent-zero guard only fires for google; anthropic 0.0 without reason is kept."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = None

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="anthropic")

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.0)
        assert result.details["samples_scored"] == 1


class TestSilentZeroHandlingPrecision:
    """ContextPrecision metric skips sessions with instructor#1658 silent-zero scores."""

    def test_silent_zero_skipped_not_averaged(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with no reason, session is skipped, not averaged."""
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = None  # instructor#1658 silent zero

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "instructor#1658" in result.details.get("skipped", "")
        assert result.details.get("silent_zero_sessions") == 1

    def test_legitimate_zero_from_google_with_reason_is_kept(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with a reason, it is a real score."""
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = "No relevant context found"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.0)
        assert result.details["samples_scored"] == 1


class TestSilentZeroHandlingRecall:
    """ContextRecall metric skips sessions with instructor#1658 silent-zero scores."""

    def test_silent_zero_skipped_not_averaged(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with no reason, session is skipped, not averaged."""
        from raki.metrics.ragas.recall import ContextRecallMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = None  # instructor#1658 silent zero

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        ground_truth = GroundTruth(
            question="How to validate?",
            reference_answer="Use pydantic",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "instructor#1658" in result.details.get("skipped", "")
        assert result.details.get("silent_zero_sessions") == 1

    def test_legitimate_zero_from_google_with_reason_is_kept(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with a reason, it is a real score."""
        from raki.metrics.ragas.recall import ContextRecallMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = "No recall found"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.0)
        assert result.details["samples_scored"] == 1


class TestSilentZeroHandlingRelevancy:
    """AnswerRelevancy metric skips sessions with instructor#1658 silent-zero scores."""

    def test_silent_zero_skipped_not_averaged(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with no reason, session is skipped, not averaged."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = None  # instructor#1658 silent zero

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "instructor#1658" in result.details.get("skipped", "")
        assert result.details.get("silent_zero_sessions") == 1

    def test_legitimate_zero_from_google_with_reason_is_kept(self, monkeypatch: pytest.MonkeyPatch):
        """When Google returns 0.0 with a reason, it is a real score."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = "The response does not address the question"

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(return_value=mock_result)

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig(llm_provider="google")

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            result = metric.compute(dataset, config)

        assert result.score == pytest.approx(0.0)
        assert result.details["samples_scored"] == 1


class TestGoogleProviderTopPRemoval:
    """Google provider must have top_p removed from model_args (mirrors Anthropic fix)."""

    def test_google_provider_pops_top_p_from_model_args(self, monkeypatch):
        """top_p must be removed from Google LLM model_args to prevent API rejection."""
        import sys

        mock_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        # LLM returned by factory has top_p in model_args
        mock_llm = MagicMock()
        mock_llm.model_args = {"temperature": 0.0, "top_p": 0.9, "max_tokens": 4096}
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

        result = llm_setup.create_ragas_llm(config)

        # top_p must have been removed
        assert "top_p" not in result.model_args

    def test_google_provider_top_p_removal_is_safe_when_absent(self, monkeypatch):
        """Removing top_p when not present must not raise KeyError."""
        import sys

        mock_client = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", mock_google_module)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        # LLM returned by factory has no top_p
        mock_llm = MagicMock()
        mock_llm.model_args = {"temperature": 0.0, "max_tokens": 4096}
        mock_factory = MagicMock(return_value=mock_llm)
        monkeypatch.setitem(sys.modules, "ragas.llms", MagicMock(llm_factory=mock_factory))

        from importlib import reload

        from raki.metrics.ragas import llm_setup

        reload(llm_setup)

        config = MetricConfig(llm_provider="google", llm_model="gemini-2.5-pro")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

        # Must not raise
        result = llm_setup.create_ragas_llm(config)
        assert result is mock_llm


# ---------------------------------------------------------------------------
# Integration tests (slow, requires LLM) — marked for selective running
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLlmSetupIntegration:
    def test_create_ragas_llm_returns_valid_object(self):
        """Requires anthropic + Vertex AI credentials. Verifies LLM setup works."""
        pytest.importorskip("ragas")
        pytest.importorskip("anthropic")
        from raki.metrics.ragas.llm_setup import create_ragas_llm

        config = MetricConfig(temperature=0.0)
        llm = create_ragas_llm(config)
        assert llm is not None

    def test_create_ragas_embeddings_returns_valid_object(self):
        """Requires ragas google_provider + credentials. Verifies embeddings setup."""
        pytest.importorskip("ragas.embeddings.google_provider")
        from raki.metrics.ragas.llm_setup import create_ragas_embeddings

        embeddings = create_ragas_embeddings()
        assert embeddings is not None


@pytest.mark.slow
class TestFaithfulnessIntegration:
    def test_faithfulness_returns_score_between_0_and_1(self):
        """Requires Ragas + LLM credentials. Scores a simple sample."""
        pytest.importorskip("ragas")
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        ground_truth = GroundTruth(
            question="How to validate input?",
            reference_answer="Use pydantic models with Field validators",
            domains=["validation"],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        metric = FaithfulnessMetric()
        result = metric.compute(dataset, config)

        assert 0.0 <= result.score <= 1.0
        assert result.details["samples_scored"] >= 1


@pytest.mark.slow
class TestAnswerRelevancyIntegration:
    def test_relevancy_returns_score_between_0_and_1(self):
        """Requires Ragas + LLM + Embeddings credentials. Scores a simple sample."""
        pytest.importorskip("ragas")
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        metric = AnswerRelevancyMetric()
        result = metric.compute(dataset, config)

        assert 0.0 <= result.score <= 1.0
        assert result.details["samples_scored"] >= 1


# ---------------------------------------------------------------------------
# Truncation tests — truncate_for_ragas and to_ragas_rows truncation
# ---------------------------------------------------------------------------


class TestTruncateForRagas:
    """Tests for the truncate_for_ragas helper function."""

    def test_short_text_unchanged(self):
        """Text shorter than the limit should be returned as-is."""
        short = "This is short text."
        assert truncate_for_ragas(short, max_chars=100) == short

    def test_text_at_limit_unchanged(self):
        """Text exactly at the limit should be returned as-is."""
        text = "a" * 100
        assert truncate_for_ragas(text, max_chars=100) == text

    def test_long_text_truncated_with_marker(self):
        """Text exceeding the limit should be truncated with [truncated] marker."""
        long_text = "word " * 5000  # ~25000 chars
        result = truncate_for_ragas(long_text, max_chars=100)
        assert len(result) <= 100 + len(" [truncated]")
        assert result.endswith("[truncated]")

    def test_truncation_at_word_boundary(self):
        """Truncation should happen at a word boundary, not mid-word."""
        # Create text with distinct words
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        result = truncate_for_ragas(text, max_chars=30)
        assert result.endswith("[truncated]")
        # The text before [truncated] should not end with a partial word
        content_part = result.replace(" [truncated]", "")
        assert not content_part.endswith("cha")  # should not cut mid-word

    def test_truncation_uses_default_max_context_chars(self):
        """Default max_chars should be MAX_CONTEXT_CHARS (1_000)."""
        assert MAX_CONTEXT_CHARS == 1_000

    def test_max_response_chars_constant(self):
        """MAX_RESPONSE_CHARS should be defined for response truncation."""
        assert MAX_RESPONSE_CHARS == 2_000

    def test_empty_string_unchanged(self):
        """Empty string should be returned as-is."""
        assert truncate_for_ragas("", max_chars=100) == ""

    def test_truncation_marker_present_only_when_truncated(self):
        """[truncated] marker should not appear when text fits."""
        text = "fits fine"
        result = truncate_for_ragas(text, max_chars=100)
        assert "[truncated]" not in result


class TestToRagasRowsTruncation:
    """Tests that to_ragas_rows truncates contexts and response."""

    def test_truncates_large_knowledge_context(self):
        """Each context chunk should be truncated to MAX_CONTEXT_CHARS."""
        huge_context = "knowledge " * 5000  # ~50000 chars per chunk
        sample = _make_sample_with_knowledge(
            knowledge_context=huge_context,
            output="short response",
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        for context in rows[0].retrieved_contexts:
            assert len(context) <= MAX_CONTEXT_CHARS + len(" [truncated]")

    def test_truncates_large_response(self):
        """The response field should be truncated to MAX_RESPONSE_CHARS."""
        huge_output = "code_line(); " * 5000  # ~65000 chars
        # Use a sample without triage to force fallback to raw output truncation
        meta = SessionMeta(
            session_id="large-resp",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output=huge_output,
            knowledge_context="normal context",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement],
            findings=[],
            events=[],
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert len(rows[0].response) <= MAX_RESPONSE_CHARS + len(" [truncated]")
        assert rows[0].response.endswith("[truncated]")

    def test_short_content_not_truncated(self):
        """Short contexts and responses should pass through unchanged."""
        # Use a sample without triage phase to test raw output passthrough
        meta = SessionMeta(
            session_id="short-test",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="short answer",
            knowledge_context="small ctx1\n---\nsmall ctx2",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement],
            findings=[],
            events=[],
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert rows[0].retrieved_contexts == ["small ctx1", "small ctx2"]
        assert rows[0].response == "short answer"
        # No truncation markers
        for context in rows[0].retrieved_contexts:
            assert "[truncated]" not in context
        assert "[truncated]" not in rows[0].response

    def test_multiple_large_context_chunks_each_truncated(self):
        """When knowledge_context has multiple chunks separated by ---, each gets truncated."""
        chunk1 = "alpha " * 5000
        chunk2 = "bravo " * 5000
        knowledge = f"{chunk1}\n---\n{chunk2}"
        sample = _make_sample_with_knowledge(
            knowledge_context=knowledge,
            output="short",
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert len(rows[0].retrieved_contexts) == 2
        for context in rows[0].retrieved_contexts:
            assert context.endswith("[truncated]")
            assert len(context) <= MAX_CONTEXT_CHARS + len(" [truncated]")


# ---------------------------------------------------------------------------
# _extract_response_summary tests
# ---------------------------------------------------------------------------


class TestExtractResponseSummary:
    """Tests for the _extract_response_summary helper function."""

    def test_prefers_triage_plan_over_raw_output(self):
        """When triage approach, plan tasks, and implement deviations exist, use them."""
        meta = SessionMeta(
            session_id="summary-1",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=0,
        )
        triage = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage output",
            output_structured={"approach": "Add input validation to the CLI"},
        )
        plan = PhaseResult(
            name="plan",
            generation=1,
            status="completed",
            output="plan output",
            output_structured={
                "tasks": [
                    {"description": "Create validator module"},
                    {"description": "Add CLI flag --validate"},
                ]
            },
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="x" * 50_000,  # Very large raw output
            output_structured={
                "deviations": ["Skipped edge case for empty input"],
                "commits": [{"message": "feat: add validation"}],
            },
            knowledge_context="some context",
        )
        sample = EvalSample(
            session=meta,
            phases=[triage, plan, implement],
            findings=[],
            events=[],
        )
        result = _extract_response_summary(sample, implement)

        # Should contain structured content, not raw output
        assert "Add input validation to the CLI" in result
        assert "Create validator module" in result
        assert "Add CLI flag --validate" in result
        assert "Skipped edge case for empty input" in result
        assert "feat: add validation" in result
        # Should NOT be the raw 50k output
        assert len(result) <= MAX_RESPONSE_CHARS

    def test_falls_back_to_truncated_output(self):
        """When no structured fields exist, fall back to truncated implement output."""
        meta = SessionMeta(
            session_id="summary-2",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="word " * 1000,  # 5000 chars
            knowledge_context="some context",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement],
            findings=[],
            events=[],
        )
        result = _extract_response_summary(sample, implement)

        # Should be the truncated raw output
        assert len(result) <= MAX_RESPONSE_CHARS + len(" [truncated]")
        assert result.endswith("[truncated]")

    def test_caps_at_max_response_chars(self):
        """Even structured summary should be capped at MAX_RESPONSE_CHARS."""
        meta = SessionMeta(
            session_id="summary-3",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=0,
        )
        triage = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage",
            output_structured={"approach": "very long approach " * 200},
        )
        plan = PhaseResult(
            name="plan",
            generation=1,
            status="completed",
            output="plan",
            output_structured={"tasks": [{"description": "task desc " * 200}]},
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="output",
            knowledge_context="ctx",
        )
        sample = EvalSample(
            session=meta,
            phases=[triage, plan, implement],
            findings=[],
            events=[],
        )
        result = _extract_response_summary(sample, implement)
        assert len(result) <= MAX_RESPONSE_CHARS + len(" [truncated]")

    def test_short_output_returned_as_is_when_no_structured(self):
        """Short implement output returned without truncation when no structured data."""
        meta = SessionMeta(
            session_id="summary-4",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="short answer",
            knowledge_context="ctx",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement],
            findings=[],
            events=[],
        )
        result = _extract_response_summary(sample, implement)
        assert result == "short answer"


# ---------------------------------------------------------------------------
# select_relevant_chunks tests
# ---------------------------------------------------------------------------


class TestSelectRelevantChunks:
    """Tests for the select_relevant_chunks function."""

    def test_returns_most_relevant_chunks(self):
        """Chunks with more keyword overlap should be ranked higher."""
        chunks = [
            DocChunk(
                text="unrelated content about cooking recipes", source_file="a.md", domain="x"
            ),
            DocChunk(
                text="validation using pydantic models and fields",
                source_file="b.md",
                domain="y",
            ),
            DocChunk(
                text="pydantic validation with Field validators for input",
                source_file="c.md",
                domain="z",
            ),
        ]
        result = select_relevant_chunks("How to validate input with pydantic?", chunks)
        # The chunk with more overlapping words should come first
        assert "pydantic validation" in result[0].text or "pydantic models" in result[0].text

    def test_respects_top_k_limit(self):
        """Should return at most top_k chunks."""
        chunks = [
            DocChunk(text=f"chunk number {idx} with keyword", source_file=f"{idx}.md", domain="d")
            for idx in range(20)
        ]
        result = select_relevant_chunks("keyword", chunks, top_k=5)
        assert len(result) == 5

    def test_default_top_k_is_max_reference_chunks(self):
        """Default top_k should be MAX_REFERENCE_CHUNKS (10)."""
        chunks = [
            DocChunk(text=f"chunk {idx} keyword", source_file=f"{idx}.md", domain="d")
            for idx in range(20)
        ]
        result = select_relevant_chunks("keyword", chunks)
        assert len(result) == MAX_REFERENCE_CHUNKS

    def test_handles_empty_query(self):
        """Empty query should return empty list (all chunks have zero overlap)."""
        chunks = [
            DocChunk(text="some content", source_file="a.md", domain="d"),
        ]
        result = select_relevant_chunks("", chunks)
        assert len(result) == 0

    def test_handles_empty_chunks(self):
        """Empty chunk list should return empty list."""
        result = select_relevant_chunks("some query", [])
        assert result == []

    def test_truncates_each_chunk_to_max_reference_chars(self):
        """Each returned chunk should be truncated to MAX_REFERENCE_CHARS."""
        long_text = "keyword " * 500  # 4000 chars
        chunks = [
            DocChunk(text=long_text, source_file="big.md", domain="d"),
        ]
        result = select_relevant_chunks("keyword", chunks)
        assert len(result) == 1
        assert len(result[0].text) <= MAX_REFERENCE_CHARS + len(" [truncated]")

    def test_fewer_chunks_than_top_k(self):
        """When fewer chunks than top_k exist, return all of them."""
        chunks = [
            DocChunk(text="keyword text", source_file="a.md", domain="d"),
            DocChunk(text="keyword stuff", source_file="b.md", domain="d"),
        ]
        result = select_relevant_chunks("keyword", chunks, top_k=10)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# to_ragas_rows context caps tests
# ---------------------------------------------------------------------------


class TestToRagasRowsContextCaps:
    """Tests for retrieved context capping in to_ragas_rows."""

    def test_caps_retrieved_contexts_at_max_context_chunks(self):
        """Retrieved contexts should be sliced to MAX_CONTEXT_CHUNKS."""
        # Create knowledge_context with many chunks
        many_chunks = "\n---\n".join(f"chunk {idx}" for idx in range(25))
        sample = _make_sample_with_knowledge(
            knowledge_context=many_chunks,
            output="short response",
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        assert len(rows[0].retrieved_contexts) == MAX_CONTEXT_CHUNKS

    def test_truncates_reference_per_chunk_not_whole_join(self):
        """When doc_chunks are provided, each should be truncated individually."""
        long_chunk_text = "validation input " * 250  # 4250 chars each, overlaps with user_input
        doc_chunks = [
            DocChunk(text=long_chunk_text, source_file=f"doc{idx}.md", domain="d")
            for idx in range(3)
        ]
        sample = _make_sample_with_knowledge(ground_truth=None)
        rows = to_ragas_rows(EvalDataset(samples=[sample]), doc_chunks=doc_chunks)
        assert len(rows) == 1
        assert rows[0].reference is not None
        # Reference should be a join of individually truncated chunks
        # Each chunk should be at most MAX_REFERENCE_CHARS (not the full 4000)
        ref_parts = rows[0].reference.split("\n\n")
        for part in ref_parts:
            assert len(part) <= MAX_REFERENCE_CHARS + len(" [truncated]")


class TestToRagasRowsResponseSummary:
    """Tests that to_ragas_rows uses _extract_response_summary."""

    def test_uses_structured_summary_instead_of_raw_output(self):
        """to_ragas_rows should call _extract_response_summary, not raw truncation."""
        meta = SessionMeta(
            session_id="resp-summary",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=0,
        )
        triage = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage out",
            output_structured={"approach": "Refactor authentication module"},
        )
        plan = PhaseResult(
            name="plan",
            generation=1,
            status="completed",
            output="plan out",
            output_structured={"tasks": [{"description": "Update auth tokens"}]},
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="x" * 50_000,
            knowledge_context="entry 1\n---\nentry 2",
        )
        sample = EvalSample(
            session=meta,
            phases=[triage, plan, implement],
            findings=[],
            events=[],
        )
        rows = to_ragas_rows(EvalDataset(samples=[sample]))
        assert len(rows) == 1
        # Should contain the structured approach, not the 50k raw output
        assert "Refactor authentication module" in rows[0].response
        assert len(rows[0].response) <= MAX_RESPONSE_CHARS + len(" [truncated]")


# ---------------------------------------------------------------------------
# Updated truncation constant tests
# ---------------------------------------------------------------------------


class TestUpdatedConstants:
    """Tests that constants have been updated to new values."""

    def test_max_context_chars(self):
        assert MAX_CONTEXT_CHARS == 1_000

    def test_max_response_chars(self):
        assert MAX_RESPONSE_CHARS == 2_000

    def test_max_context_chunks(self):
        assert MAX_CONTEXT_CHUNKS == 10

    def test_max_reference_chunks(self):
        assert MAX_REFERENCE_CHUNKS == 10

    def test_max_reference_chars(self):
        assert MAX_REFERENCE_CHARS == 1_000


# ---------------------------------------------------------------------------
# max_tokens error handling tests
# ---------------------------------------------------------------------------


class TestMaxTokensErrorHandling:
    """Tests that max_tokens errors return score=None instead of score=0.0."""

    def _make_max_tokens_error(self) -> Exception:
        """Create an exception that looks like a max_tokens error."""
        return RuntimeError("max_tokens: output token limit reached")

    def test_faithfulness_returns_none_on_max_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Faithfulness should return score=None on max_tokens errors."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=self._make_max_tokens_error())

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "max_tokens" in result.details.get("skipped", "")

    def test_relevancy_returns_none_on_max_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """AnswerRelevancy should return score=None on max_tokens errors."""
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=self._make_max_tokens_error())

        mock_relevancy_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_relevancy_class, "AnswerRelevancy")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with (
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_llm",
                return_value=MagicMock(),
            ),
            patch(
                "raki.metrics.ragas.relevancy.create_ragas_embeddings",
                return_value=MagicMock(),
            ),
        ):
            metric = AnswerRelevancyMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "max_tokens" in result.details.get("skipped", "")

    def test_precision_returns_none_on_max_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """ContextPrecision should return score=None on max_tokens errors."""
        from raki.metrics.ragas.precision import ContextPrecisionMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=self._make_max_tokens_error())

        mock_precision_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_precision_class, "ContextPrecisionWithReference")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.precision.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextPrecisionMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "max_tokens" in result.details.get("skipped", "")

    def test_recall_returns_none_on_max_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """ContextRecall should return score=None on max_tokens errors."""
        from raki.metrics.ragas.recall import ContextRecallMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=self._make_max_tokens_error())

        mock_recall_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_recall_class, "ContextRecall")

        ground_truth = GroundTruth(
            question="Q?",
            reference_answer="A",
            domains=[],
        )
        sample = _make_sample_with_knowledge(ground_truth=ground_truth)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.recall.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = ContextRecallMetric()
            result = metric.compute(dataset, config)

        assert result.score is None
        assert "max_tokens" in result.details.get("skipped", "")

    def test_non_max_tokens_error_still_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-max_tokens errors should still result in score=0.0 (existing behavior)."""
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        monkeypatch.chdir(tmp_path)

        mock_metric_instance = MagicMock()
        mock_metric_instance.ascore = AsyncMock(side_effect=RuntimeError("Connection refused"))

        mock_faithfulness_class = MagicMock(return_value=mock_metric_instance)
        _install_ragas_mock(monkeypatch, mock_faithfulness_class, "Faithfulness")

        sample = _make_sample_with_knowledge()
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()

        with patch(
            "raki.metrics.ragas.faithfulness.create_ragas_llm",
            return_value=MagicMock(),
        ):
            metric = FaithfulnessMetric()
            result = metric.compute(dataset, config)

        assert result.score == 0.0
        assert result.details["samples_scored"] == 0


# ---------------------------------------------------------------------------
# Edge-case tests for _extract_response_summary and select_relevant_chunks
# ---------------------------------------------------------------------------


class TestExtractResponseSummaryEdgeCases:
    """Edge-case tests for _extract_response_summary."""

    def _make_sample_with_phases(self, phases: list[PhaseResult]) -> EvalSample:
        meta = SessionMeta(
            session_id="edge-case",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=len(phases),
            rework_cycles=0,
        )
        return EvalSample(
            session=meta,
            phases=phases,
            findings=[],
            events=[],
        )

    def test_extract_summary_missing_approach_key(self):
        """Triage with output_structured that has no 'approach' key should not crash."""
        triage = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage output",
            output_structured={"severity": "high"},
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="fallback output",
            knowledge_context="ctx",
        )
        sample = self._make_sample_with_phases([triage, implement])
        result = _extract_response_summary(sample, implement)
        # Should fall back to raw output since no approach was found
        assert result == "fallback output"

    def test_extract_summary_non_string_approach(self):
        """Triage with approach as a dict should skip it."""
        triage = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage output",
            output_structured={"approach": {"nested": "value"}},
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="fallback output",
            knowledge_context="ctx",
        )
        sample = self._make_sample_with_phases([triage, implement])
        result = _extract_response_summary(sample, implement)
        # approach is a dict, not a string, so it should be skipped
        assert result == "fallback output"

    def test_extract_summary_plan_tasks_non_dict_entries(self):
        """Plan with tasks as list of strings should skip non-dict entries."""
        triage = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage output",
            output_structured={"approach": "Fix the bug"},
        )
        plan = PhaseResult(
            name="plan",
            generation=1,
            status="completed",
            output="plan output",
            output_structured={"tasks": ["string task 1", "string task 2"]},
        )
        implement = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="impl output",
            knowledge_context="ctx",
        )
        sample = self._make_sample_with_phases([triage, plan, implement])
        result = _extract_response_summary(sample, implement)
        # Should have approach but no tasks (strings are skipped, only dicts accepted)
        assert "Fix the bug" in result
        assert "string task 1" not in result

    def test_extract_summary_latest_generation_only(self):
        """Session with multiple triage phases should use only the latest generation."""
        triage_gen1 = PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="old triage",
            output_structured={"approach": "Old approach from gen 1"},
        )
        triage_gen2 = PhaseResult(
            name="triage",
            generation=2,
            status="completed",
            output="new triage",
            output_structured={"approach": "New approach from gen 2"},
        )
        implement = PhaseResult(
            name="implement",
            generation=2,
            status="completed",
            output="impl output",
            knowledge_context="ctx",
        )
        sample = self._make_sample_with_phases([triage_gen1, triage_gen2, implement])
        result = _extract_response_summary(sample, implement)
        # Should use only the latest generation (gen 2)
        assert "New approach from gen 2" in result
        assert "Old approach from gen 1" not in result


class TestSelectRelevantChunksEdgeCases:
    """Edge-case tests for select_relevant_chunks."""

    def test_select_relevant_chunks_zero_overlap_returns_empty(self):
        """Query with no overlapping words should return empty list."""
        chunks = [
            DocChunk(text="alpha bravo charlie", source_file="a.md", domain="d"),
            DocChunk(text="delta echo foxtrot", source_file="b.md", domain="d"),
        ]
        result = select_relevant_chunks("xylophone zebra quantum", chunks)
        assert result == []


# ---------------------------------------------------------------------------
# ScoringLoop tests — shared scoring loop extracted from the 4 Ragas metrics
# ---------------------------------------------------------------------------


class TestScoringState:
    """Unit tests for ScoringState dataclass."""

    def test_default_state_is_empty(self):
        from raki.metrics.ragas._scoring_loop import ScoringState

        state = ScoringState()
        assert state.scores == []
        assert state.sample_scores == {}
        assert state.max_tokens_failures == []
        assert state.silent_zero_failures == []

    def test_mean_score_empty_returns_zero(self):
        from raki.metrics.ragas._scoring_loop import ScoringState

        state = ScoringState()
        assert state.mean_score == 0.0

    def test_mean_score_single_value(self):
        from raki.metrics.ragas._scoring_loop import ScoringState

        state = ScoringState(scores=[0.8])
        assert state.mean_score == pytest.approx(0.8)

    def test_mean_score_averages_multiple_values(self):
        from raki.metrics.ragas._scoring_loop import ScoringState

        state = ScoringState(scores=[0.6, 0.8, 1.0])
        assert state.mean_score == pytest.approx(0.8)


class TestScoreRows:
    """Tests for the score_rows() coroutine."""

    def _make_row(self, session_id: str = "s1") -> RagasRow:
        return RagasRow(
            session_id=session_id,
            user_input="How to validate?",
            retrieved_contexts=["ctx"],
            response="Use pydantic",
            reference=None,
        )

    def test_successful_scoring_accumulates_scores(self):
        """score_rows should append scores and sample_scores for each row."""
        import asyncio

        from raki.metrics.ragas._scoring_loop import ScoringState, score_rows

        async def fake_score_fn(row: RagasRow) -> float:
            return 0.85

        rows = [self._make_row("s1"), self._make_row("s2")]
        state: ScoringState = asyncio.run(
            score_rows(
                rows=rows,
                score_fn=fake_score_fn,
                metric_name="test_metric",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=None,
            )
        )

        assert len(state.scores) == 2
        assert state.sample_scores == {"s1": pytest.approx(0.85), "s2": pytest.approx(0.85)}
        assert state.max_tokens_failures == []
        assert state.silent_zero_failures == []

    def test_handles_structured_result_with_value_attribute(self):
        """score_rows handles result objects with .value and .reason attributes."""
        import asyncio
        from unittest.mock import MagicMock

        from raki.metrics.ragas._scoring_loop import score_rows

        mock_result = MagicMock()
        mock_result.value = 0.72
        mock_result.reason = "Good"

        async def fake_score_fn(row: RagasRow):
            return mock_result

        rows = [self._make_row("s1")]
        state = asyncio.run(
            score_rows(
                rows=rows,
                score_fn=fake_score_fn,
                metric_name="test_metric",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=None,
            )
        )

        assert state.scores == [pytest.approx(0.72)]
        assert state.sample_scores == {"s1": pytest.approx(0.72)}

    def test_max_tokens_error_recorded_in_failures(self):
        """Exceptions containing 'max_tokens' should go into max_tokens_failures."""
        import asyncio

        from raki.metrics.ragas._scoring_loop import score_rows

        async def failing_score_fn(row: RagasRow) -> float:
            raise RuntimeError("max_tokens limit exceeded")

        rows = [self._make_row("s1")]
        state = asyncio.run(
            score_rows(
                rows=rows,
                score_fn=failing_score_fn,
                metric_name="test_metric",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=None,
            )
        )

        assert state.scores == []
        assert state.max_tokens_failures == ["s1"]
        assert state.silent_zero_failures == []

    def test_instructor_silent_zero_recorded_in_failures(self):
        """InstructorSilentZeroError should go into silent_zero_failures."""
        import asyncio
        from unittest.mock import MagicMock

        from raki.metrics.ragas._scoring_loop import score_rows

        # A result that triggers silent-zero detection (value=0.0, reason=None, provider=google)
        mock_result = MagicMock()
        mock_result.value = 0.0
        mock_result.reason = None

        async def fake_score_fn(row: RagasRow):
            return mock_result

        rows = [self._make_row("s1")]
        state = asyncio.run(
            score_rows(
                rows=rows,
                score_fn=fake_score_fn,
                metric_name="test_metric",
                llm_provider="google",  # only detected for google
                batch_size=4,
                judge_logger=None,
            )
        )

        assert state.scores == []
        assert state.silent_zero_failures == ["s1"]
        assert state.max_tokens_failures == []

    def test_generic_exception_does_not_populate_failure_lists(self):
        """Non-classified exceptions should leave both failure lists empty."""
        import asyncio

        from raki.metrics.ragas._scoring_loop import score_rows

        async def failing_score_fn(row: RagasRow) -> float:
            raise ValueError("Something else went wrong")

        rows = [self._make_row("s1")]
        state = asyncio.run(
            score_rows(
                rows=rows,
                score_fn=failing_score_fn,
                metric_name="test_metric",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=None,
            )
        )

        assert state.scores == []
        assert state.max_tokens_failures == []
        assert state.silent_zero_failures == []

    def test_mixed_success_and_failure(self):
        """score_rows should record only successful scores, not failed ones."""
        import asyncio

        from raki.metrics.ragas._scoring_loop import score_rows

        call_count = 0

        async def alternating_score_fn(row: RagasRow) -> float:
            nonlocal call_count
            call_count += 1
            if row.session_id == "s2":
                raise RuntimeError("max_tokens error for s2")
            return 0.9

        rows = [self._make_row("s1"), self._make_row("s2")]
        state = asyncio.run(
            score_rows(
                rows=rows,
                score_fn=alternating_score_fn,
                metric_name="test_metric",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=None,
            )
        )

        assert state.scores == [pytest.approx(0.9)]
        assert state.sample_scores == {"s1": pytest.approx(0.9)}
        assert state.max_tokens_failures == ["s2"]

    def test_judge_logger_called_on_success(self, tmp_path: Path):
        """score_rows should call judge_logger.log for successful scores."""
        import asyncio

        from raki.metrics.ragas._scoring_loop import score_rows

        judge_logger = JudgeLogger(tmp_path / "judge.jsonl", project_root=tmp_path)

        async def fake_score_fn(row: RagasRow) -> float:
            return 0.75

        rows = [self._make_row("s1")]
        asyncio.run(
            score_rows(
                rows=rows,
                score_fn=fake_score_fn,
                metric_name="faithfulness",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=judge_logger,
            )
        )

        import json

        entries = [
            json.loads(line) for line in (tmp_path / "judge.jsonl").read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["metric"] == "faithfulness"
        assert entries[0]["score"] == 0.75

    def test_judge_logger_called_on_failure(self, tmp_path: Path):
        """score_rows should log score=-1.0 to judge_logger on failure."""
        import asyncio

        from raki.metrics.ragas._scoring_loop import score_rows

        judge_logger = JudgeLogger(tmp_path / "judge.jsonl", project_root=tmp_path)

        async def failing_score_fn(row: RagasRow) -> float:
            raise RuntimeError("LLM error")

        rows = [self._make_row("s1")]
        asyncio.run(
            score_rows(
                rows=rows,
                score_fn=failing_score_fn,
                metric_name="faithfulness",
                llm_provider="vertex-anthropic",
                batch_size=4,
                judge_logger=judge_logger,
            )
        )

        import json

        entries = [
            json.loads(line) for line in (tmp_path / "judge.jsonl").read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["score"] == -1.0
        assert "LLM error" in entries[0]["reason"]


class TestBuildMaxTokensResult:
    """Tests for build_max_tokens_result() helper."""

    def test_returns_none_when_scores_exist(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, build_max_tokens_result

        state = ScoringState(scores=[0.8], max_tokens_failures=["s1"])
        result = build_max_tokens_result("test_metric", state)
        assert result is None

    def test_returns_none_when_no_failures(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, build_max_tokens_result

        state = ScoringState(scores=[], max_tokens_failures=[])
        result = build_max_tokens_result("test_metric", state)
        assert result is None

    def test_returns_metric_result_when_all_max_tokens(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, build_max_tokens_result

        state = ScoringState(scores=[], max_tokens_failures=["s1", "s2"])
        result = build_max_tokens_result("faithfulness", state)
        assert result is not None
        assert result.name == "faithfulness"
        assert result.score is None
        assert result.details["skipped"] == "max_tokens: all sessions exceeded output token limit"
        assert result.details["max_tokens_sessions"] == 2

    def test_returns_none_when_only_silent_zero_failures(self):
        """build_max_tokens_result should not fire when only silent-zero failures exist."""
        from raki.metrics.ragas._scoring_loop import ScoringState, build_max_tokens_result

        state = ScoringState(scores=[], max_tokens_failures=[], silent_zero_failures=["s1"])
        result = build_max_tokens_result("test_metric", state)
        assert result is None


class TestBuildSilentZeroResult:
    """Tests for build_silent_zero_result() helper."""

    def test_returns_none_when_scores_exist(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, build_silent_zero_result

        state = ScoringState(scores=[0.7], silent_zero_failures=["s1"])
        result = build_silent_zero_result("test_metric", state)
        assert result is None

    def test_returns_none_when_no_failures(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, build_silent_zero_result

        state = ScoringState(scores=[], silent_zero_failures=[])
        result = build_silent_zero_result("test_metric", state)
        assert result is None

    def test_returns_metric_result_when_all_silent_zero(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, build_silent_zero_result

        state = ScoringState(scores=[], silent_zero_failures=["s1", "s2", "s3"])
        result = build_silent_zero_result("context_precision", state)
        assert result is not None
        assert result.name == "context_precision"
        assert result.score is None
        assert "instructor#1658" in result.details["skipped"]
        assert result.details["silent_zero_sessions"] == 3

    def test_returns_none_when_only_max_tokens_failures(self):
        """build_silent_zero_result should not fire when only max_tokens failures exist."""
        from raki.metrics.ragas._scoring_loop import ScoringState, build_silent_zero_result

        state = ScoringState(scores=[], max_tokens_failures=["s1"], silent_zero_failures=[])
        result = build_silent_zero_result("test_metric", state)
        assert result is None


class TestEnrichDetailsWithFailures:
    """Tests for enrich_details_with_failures() helper."""

    def test_no_failures_leaves_details_unchanged(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, enrich_details_with_failures

        state = ScoringState()
        details: dict = {"samples_scored": 5}
        enrich_details_with_failures(details, state)
        assert details == {"samples_scored": 5}

    def test_max_tokens_failures_adds_count(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, enrich_details_with_failures

        state = ScoringState(max_tokens_failures=["s1", "s2"])
        details: dict = {}
        enrich_details_with_failures(details, state)
        assert details["max_tokens_sessions"] == 2
        assert "silent_zero_sessions" not in details

    def test_silent_zero_failures_adds_count_and_warning(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, enrich_details_with_failures

        state = ScoringState(silent_zero_failures=["s1", "s2", "s3"])
        details: dict = {}
        enrich_details_with_failures(details, state)
        assert details["silent_zero_sessions"] == 3
        assert "silent_zero_warning" in details
        assert "instructor#1658" in details["silent_zero_warning"]
        assert "3 session(s)" in details["silent_zero_warning"]

    def test_both_failure_types_adds_all_keys(self):
        from raki.metrics.ragas._scoring_loop import ScoringState, enrich_details_with_failures

        state = ScoringState(max_tokens_failures=["s1"], silent_zero_failures=["s2"])
        details: dict = {}
        enrich_details_with_failures(details, state)
        assert details["max_tokens_sessions"] == 1
        assert details["silent_zero_sessions"] == 1
        assert "silent_zero_warning" in details
