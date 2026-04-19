"""Tests for operational metrics: verify rate, rework cycles, severity, cost."""

from datetime import datetime, timezone

import pytest

from conftest import make_dataset, make_sample
from raki.metrics.engine import MetricsEngine
from raki.metrics.operational.cost import CostEfficiency
from raki.metrics.operational.rework import ReworkCycles
from raki.metrics.operational.severity import ReviewSeverityDistribution
from raki.metrics.operational.verify_rate import FirstPassVerifyRate
from raki.metrics.protocol import MetricConfig
from raki.model import (
    EvalDataset,
    EvalSample,
    PhaseResult,
    ReviewFinding,
    SessionMeta,
)
from raki.model.report import SampleResult


# --- FirstPassVerifyRate ---


def test_first_pass_verify_rate_all_pass():
    dataset = make_dataset(
        make_sample("1", verify_gen=1),
        make_sample("2", verify_gen=1),
    )
    result = FirstPassVerifyRate().compute(dataset, MetricConfig())
    assert result.score == 1.0
    assert result.details["passed"] == 2
    assert result.details["total"] == 2


def test_first_pass_verify_rate_mixed():
    dataset = make_dataset(
        make_sample("1", verify_gen=1),
        make_sample("2", verify_gen=3),
        make_sample("3", verify_gen=1),
    )
    result = FirstPassVerifyRate().compute(dataset, MetricConfig())
    assert abs(result.score - 2 / 3) < 0.01
    assert result.sample_scores["2"] == 0.0


def test_first_pass_verify_rate_no_verify_phase():
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
    dataset = make_dataset(sample)
    result = FirstPassVerifyRate().compute(dataset, MetricConfig())
    assert result.details["total"] == 0
    assert result.score == 0.0


def test_first_pass_verify_rate_gen1_failed():
    """Verify gen=1, status='failed' should score 0.0."""
    dataset = make_dataset(
        make_sample("1", verify_gen=1, verify_status="failed"),
    )
    result = FirstPassVerifyRate().compute(dataset, MetricConfig())
    assert result.score == 0.0
    assert result.sample_scores["1"] == 0.0


def test_first_pass_verify_rate_properties():
    metric = FirstPassVerifyRate()
    assert metric.name == "first_pass_verify_rate"
    assert metric.requires_ground_truth is False
    assert metric.requires_llm is False
    assert metric.higher_is_better is True
    assert metric.display_format == "percent"
    assert metric.display_name == "Verify rate"


# --- ReworkCycles ---


def test_rework_cycles():
    dataset = make_dataset(
        make_sample("1", rework_cycles=0),
        make_sample("2", rework_cycles=2),
        make_sample("3", rework_cycles=1),
    )
    result = ReworkCycles().compute(dataset, MetricConfig())
    assert abs(result.score - 1.0) < 0.01
    assert result.sample_scores["2"] == 2.0


def test_rework_cycles_empty_dataset():
    """ReworkCycles with empty dataset should score 0.0."""
    dataset = EvalDataset(samples=[])
    result = ReworkCycles().compute(dataset, MetricConfig())
    assert result.score == 0.0


def test_rework_cycles_properties():
    metric = ReworkCycles()
    assert metric.name == "rework_cycles"
    assert metric.requires_ground_truth is False
    assert metric.requires_llm is False
    assert metric.higher_is_better is False
    assert metric.display_format == "count"
    assert metric.display_name == "Rework cycles"


# --- ReviewSeverityDistribution ---


def test_review_severity_distribution():
    findings = [
        ReviewFinding(reviewer="go", severity="critical", issue="panic"),
        ReviewFinding(reviewer="go", severity="major", issue="leak"),
        ReviewFinding(reviewer="ai", severity="minor", issue="nit"),
        ReviewFinding(reviewer="ai", severity="minor", issue="style"),
    ]
    dataset = make_dataset(make_sample("1", findings=findings))
    result = ReviewSeverityDistribution().compute(dataset, MetricConfig())
    assert result.details["critical"] == 1
    assert result.details["major"] == 1
    assert result.details["minor"] == 2
    assert result.details["total"] == 4
    # weighted: (3*1 + 2*1 + 1*2) / (3*4) = 7/12 ~ 0.583
    # score = 1.0 - 0.583 ~ 0.417
    assert 0.4 < result.score < 0.45


def test_review_severity_distribution_no_findings():
    dataset = make_dataset(make_sample("1", findings=[]))
    result = ReviewSeverityDistribution().compute(dataset, MetricConfig())
    assert result.score == 1.0
    assert result.details["total"] == 0


def test_review_severity_distribution_properties():
    metric = ReviewSeverityDistribution()
    assert metric.name == "review_severity_distribution"
    assert metric.requires_ground_truth is False
    assert metric.requires_llm is False
    assert metric.higher_is_better is True
    assert metric.display_format == "score"
    assert metric.display_name == "Severity score"


# --- CostEfficiency ---


def test_cost_efficiency():
    dataset = make_dataset(
        make_sample("1", cost=10.0),
        make_sample("2", cost=20.0),
        make_sample("3", cost=15.0),
    )
    result = CostEfficiency().compute(dataset, MetricConfig())
    assert abs(result.score - 15.0) < 0.01
    assert result.sample_scores["2"] == 20.0


def test_cost_efficiency_handles_missing_cost():
    meta = SessionMeta(
        session_id="x",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
        total_cost_usd=None,
    )
    sample = EvalSample(session=meta, phases=[], findings=[], events=[])
    dataset = make_dataset(sample)
    result = CostEfficiency().compute(dataset, MetricConfig())
    assert result.details["sessions_with_cost"] == 0
    assert result.score == 0.0


def test_cost_efficiency_properties():
    metric = CostEfficiency()
    assert metric.name == "cost_efficiency"
    assert metric.requires_ground_truth is False
    assert metric.requires_llm is False
    assert metric.higher_is_better is False
    assert metric.display_format == "currency"
    assert metric.display_name == "Cost / session"


# --- MetricsEngine ---


def test_engine_skips_llm_metrics():
    """MetricsEngine.run() with skip_llm=True should skip metrics requiring LLM."""
    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    report = engine.run(dataset, skip_llm=True)
    # Both operational metrics should be included (neither requires LLM)
    assert "first_pass_verify_rate" in report.aggregate_scores
    assert "rework_cycles" in report.aggregate_scores


def test_engine_skips_ground_truth_metrics():
    """MetricsEngine.run() with skip_ground_truth=True should skip metrics requiring ground truth."""
    metrics = [FirstPassVerifyRate(), CostEfficiency()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    report = engine.run(dataset, skip_ground_truth=True)
    # Both operational metrics should be included (neither requires ground truth)
    assert "first_pass_verify_rate" in report.aggregate_scores
    assert "cost_efficiency" in report.aggregate_scores


def test_engine_run_single():
    """MetricsEngine.run_single() runs only the named metric."""
    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    result = engine.run_single("rework_cycles", dataset)
    assert result.name == "rework_cycles"

    with pytest.raises(ValueError, match="Unknown metric"):
        engine.run_single("nonexistent_metric", dataset)


def test_engine_sample_results_populated():
    """engine.run() should populate report.sample_results with one SampleResult per session."""
    sample_a = make_sample("session-a", rework_cycles=0, cost=10.0)
    sample_b = make_sample("session-b", rework_cycles=2, cost=20.0)
    dataset = make_dataset(sample_a, sample_b)

    metrics = [FirstPassVerifyRate(), ReworkCycles(), CostEfficiency()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    assert len(report.sample_results) == 2
    for sample_result in report.sample_results:
        assert isinstance(sample_result, SampleResult)


def test_engine_sample_results_correct_session_ids():
    """Each SampleResult should reference the correct session_id via its EvalSample."""
    sample_a = make_sample("session-a", rework_cycles=0)
    sample_b = make_sample("session-b", rework_cycles=3)
    dataset = make_dataset(sample_a, sample_b)

    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    session_ids = {sr.sample.session.session_id for sr in report.sample_results}
    assert session_ids == {"session-a", "session-b"}


def test_engine_sample_results_contain_metric_scores():
    """Each SampleResult should contain per-metric scores for that session."""
    sample_a = make_sample("session-a", rework_cycles=0, cost=10.0)
    sample_b = make_sample("session-b", rework_cycles=2, cost=20.0)
    dataset = make_dataset(sample_a, sample_b)

    metrics = [FirstPassVerifyRate(), ReworkCycles(), CostEfficiency()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    # Find sample_result for session-b
    result_b = next(
        sr for sr in report.sample_results if sr.sample.session.session_id == "session-b"
    )
    score_names = {metric_score.name for metric_score in result_b.scores}
    assert score_names == {"first_pass_verify_rate", "rework_cycles", "cost_efficiency"}

    # Verify that sample-level scores match expected values
    rework_score = next(ms for ms in result_b.scores if ms.name == "rework_cycles")
    assert rework_score.score == 2.0

    cost_score = next(ms for ms in result_b.scores if ms.name == "cost_efficiency")
    assert cost_score.score == 20.0


def test_engine_sample_results_with_skipped_metrics():
    """SampleResult should only contain scores for metrics that were not skipped."""
    dataset = make_dataset(make_sample("session-a"))

    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset, skip_llm=True)

    # Neither metric requires LLM, so both should appear
    assert len(report.sample_results) == 1
    score_names = {ms.name for ms in report.sample_results[0].scores}
    assert score_names == {"first_pass_verify_rate", "rework_cycles"}


def test_engine_sample_results_empty_dataset():
    """engine.run() with an empty dataset should produce empty sample_results."""
    dataset = EvalDataset(samples=[])
    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    assert report.sample_results == []


def test_engine_sample_results_excludes_metrics_without_sample_scores():
    """Metrics that don't populate sample_scores (like ReviewSeverityDistribution)
    should appear in aggregate_scores but NOT in sample_results[].scores.

    This is by design: aggregate-only metrics compute a dataset-wide value and
    have no meaningful per-session breakdown, so _build_sample_results excludes
    them from the per-session drill-down.
    """
    findings = [
        ReviewFinding(reviewer="ai", severity="critical", issue="panic"),
        ReviewFinding(reviewer="ai", severity="minor", issue="nit"),
    ]
    dataset = make_dataset(
        make_sample("session-a", rework_cycles=1, findings=findings),
    )

    metrics = [ReworkCycles(), ReviewSeverityDistribution()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    # Both metrics must appear in aggregate_scores
    assert "rework_cycles" in report.aggregate_scores
    assert "review_severity_distribution" in report.aggregate_scores

    # Only ReworkCycles populates sample_scores, so ReviewSeverityDistribution
    # must be absent from the per-session drill-down
    assert len(report.sample_results) == 1
    score_names = {ms.name for ms in report.sample_results[0].scores}
    assert "rework_cycles" in score_names
    assert "review_severity_distribution" not in score_names


# --- KnowledgeRetrievalMissRate ---


def test_knowledge_miss_rate_no_rework():
    """Sessions with zero rework should produce score 0.0."""
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    dataset = make_dataset(
        make_sample("1", rework_cycles=0),
        make_sample("2", rework_cycles=0),
    )
    result = KnowledgeRetrievalMissRate().compute(dataset, MetricConfig())
    assert result.score == 0.0
    assert result.details["total_rework_findings"] == 0
    assert result.details["retrieval_gaps"] == 0
    assert result.details["capability_gaps"] == 0


def test_knowledge_miss_rate_retrieval_gap():
    """Finding with no related knowledge should be classified as retrieval_gap."""
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    meta = SessionMeta(
        session_id="rework-1",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=3,
        rework_cycles=1,
    )
    implement_phase = PhaseResult(
        name="implement",
        generation=1,
        status="completed",
        output="done",
        knowledge_context="information about database schemas and migrations",
    )
    finding = ReviewFinding(
        reviewer="ai-review",
        severity="critical",
        issue="Missing authentication check on the API endpoint",
    )
    sample = EvalSample(
        session=meta,
        phases=[implement_phase],
        findings=[finding],
        events=[],
    )
    dataset = make_dataset(sample)
    result = KnowledgeRetrievalMissRate().compute(dataset, MetricConfig())

    assert result.score == 1.0  # all findings are retrieval gaps
    assert result.details["retrieval_gaps"] == 1
    assert result.details["capability_gaps"] == 0
    assert result.details["total_rework_findings"] == 1


def test_knowledge_miss_rate_capability_gap():
    """Finding with related knowledge present should be classified as capability_gap."""
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    meta = SessionMeta(
        session_id="rework-2",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=3,
        rework_cycles=1,
    )
    implement_phase = PhaseResult(
        name="implement",
        generation=1,
        status="completed",
        output="done",
        knowledge_context="Authentication must validate tokens before processing requests",
    )
    finding = ReviewFinding(
        reviewer="ai-review",
        severity="major",
        issue="Missing authentication check on the endpoint",
    )
    sample = EvalSample(
        session=meta,
        phases=[implement_phase],
        findings=[finding],
        events=[],
    )
    dataset = make_dataset(sample)
    result = KnowledgeRetrievalMissRate().compute(dataset, MetricConfig())

    assert result.score == 0.0  # no retrieval gaps, all capability gaps
    assert result.details["retrieval_gaps"] == 0
    assert result.details["capability_gaps"] == 1
    assert result.details["total_rework_findings"] == 1


def test_knowledge_miss_rate_mixed_gaps():
    """Mix of retrieval and capability gaps should produce correct ratio."""
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    meta = SessionMeta(
        session_id="rework-3",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=3,
        rework_cycles=1,
    )
    implement_phase = PhaseResult(
        name="implement",
        generation=1,
        status="completed",
        output="done",
        knowledge_context="Authentication tokens must be validated before processing",
    )
    finding_capability = ReviewFinding(
        reviewer="ai-review",
        severity="critical",
        issue="Missing authentication token validation",
    )
    finding_retrieval = ReviewFinding(
        reviewer="ai-review",
        severity="major",
        issue="Database connection pooling not configured properly",
    )
    sample = EvalSample(
        session=meta,
        phases=[implement_phase],
        findings=[finding_capability, finding_retrieval],
        events=[],
    )
    dataset = make_dataset(sample)
    result = KnowledgeRetrievalMissRate().compute(dataset, MetricConfig())

    assert result.details["total_rework_findings"] == 2
    assert result.details["capability_gaps"] == 1
    assert result.details["retrieval_gaps"] == 1
    assert result.score == pytest.approx(0.5)  # 1 retrieval / 2 total


def test_knowledge_miss_rate_ignores_minor_findings():
    """Minor findings should not be counted in the miss rate."""
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    meta = SessionMeta(
        session_id="minor-only",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=3,
        rework_cycles=1,
    )
    implement_phase = PhaseResult(
        name="implement",
        generation=1,
        status="completed",
        output="done",
    )
    finding = ReviewFinding(
        reviewer="ai-review",
        severity="minor",
        issue="Style nit: use snake_case",
    )
    sample = EvalSample(
        session=meta,
        phases=[implement_phase],
        findings=[finding],
        events=[],
    )
    dataset = make_dataset(sample)
    result = KnowledgeRetrievalMissRate().compute(dataset, MetricConfig())

    assert result.score == 0.0
    assert result.details["total_rework_findings"] == 0


def test_knowledge_miss_rate_properties():
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    metric = KnowledgeRetrievalMissRate()
    assert metric.name == "knowledge_retrieval_miss_rate"
    assert metric.requires_ground_truth is False
    assert metric.requires_llm is False
    assert metric.higher_is_better is False
    assert metric.display_format == "score"
    assert metric.display_name == "Knowledge miss rate"


def test_knowledge_miss_rate_uses_session_phase_fallback():
    """Should find knowledge_context from 'session' phase when 'implement' is absent."""
    from raki.metrics.operational.knowledge_miss_rate import KnowledgeRetrievalMissRate

    meta = SessionMeta(
        session_id="session-phase",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=2,
        rework_cycles=1,
    )
    session_phase = PhaseResult(
        name="session",
        generation=1,
        status="completed",
        output="done",
        knowledge_context="information about authentication and authorization patterns",
    )
    finding = ReviewFinding(
        reviewer="ai-review",
        severity="critical",
        issue="Missing authentication check",
    )
    sample = EvalSample(
        session=meta,
        phases=[session_phase],
        findings=[finding],
        events=[],
    )
    dataset = make_dataset(sample)
    result = KnowledgeRetrievalMissRate().compute(dataset, MetricConfig())

    # "authentication" appears in knowledge_context, so it's a capability gap
    assert result.details["capability_gaps"] == 1
    assert result.details["retrieval_gaps"] == 0
