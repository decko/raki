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
from raki.metrics.ragas.adapter import RagasRow, to_ragas_rows
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
        assert rows[0].response == "The answer based on knowledge"
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
        assert result.score == 0.0
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
        assert result.score == 0.0
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
    def test_skips_without_samples(self):
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric()
        sample = _make_sample_with_knowledge(knowledge_context=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score == 0.0
        assert result.details.get("skipped") == "no samples"

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
        assert call_kwargs["response"] == "The answer based on knowledge"
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
    def test_skips_without_samples(self):
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        sample = _make_sample_with_knowledge(knowledge_context=None)
        dataset = EvalDataset(samples=[sample])
        config = MetricConfig()
        result = metric.compute(dataset, config)
        assert result.score == 0.0
        assert result.details.get("skipped") == "no samples"

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
        assert call_kwargs["response"] == "The answer based on knowledge"
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


class TestCreateRagasEmbeddings:
    def test_uses_vertex_ai_embeddings(self, monkeypatch: pytest.MonkeyPatch):
        """create_ragas_embeddings() should use VertexAIEmbeddings, not OpenAI default."""
        import sys

        mock_vertex_instance = MagicMock()
        mock_vertex_class = MagicMock(return_value=mock_vertex_instance)

        mock_langchain_vertex = MagicMock()
        mock_langchain_vertex.VertexAIEmbeddings = mock_vertex_class

        monkeypatch.setitem(sys.modules, "langchain_google_vertexai", mock_langchain_vertex)

        from raki.metrics.ragas.llm_setup import create_ragas_embeddings

        result = create_ragas_embeddings()
        mock_vertex_class.assert_called_once_with(model_name="text-embedding-005")
        assert result is mock_vertex_instance

    def test_does_not_use_openai_default(self, monkeypatch: pytest.MonkeyPatch):
        """Verify we don't call the OpenAI-defaulting embedding_factory()."""
        import sys

        mock_embedding_factory = MagicMock()
        mock_embeddings_module = MagicMock()
        mock_embeddings_module.embedding_factory = mock_embedding_factory

        mock_vertex_class = MagicMock(return_value=MagicMock())
        mock_langchain_vertex = MagicMock()
        mock_langchain_vertex.VertexAIEmbeddings = mock_vertex_class

        monkeypatch.setitem(sys.modules, "ragas", MagicMock())
        monkeypatch.setitem(sys.modules, "ragas.embeddings", mock_embeddings_module)
        monkeypatch.setitem(sys.modules, "langchain_google_vertexai", mock_langchain_vertex)

        from raki.metrics.ragas.llm_setup import create_ragas_embeddings

        create_ragas_embeddings()
        # embedding_factory (which defaults to OpenAI) should NOT be called
        mock_embedding_factory.assert_not_called()


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
        """Requires langchain-google-vertexai + credentials. Verifies embeddings setup."""
        pytest.importorskip("langchain_google_vertexai")
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
