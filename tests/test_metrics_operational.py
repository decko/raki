"""Tests for operational metrics: first-pass success rate, rework cycles, severity, cost."""

from datetime import datetime, timezone

import pytest

from conftest import make_dataset, make_sample
from raki.metrics.engine import MetricsEngine
from raki.metrics.operational.cost import CostEfficiency
from raki.metrics.operational.rework import ReworkCycles
from raki.metrics.operational.severity import ReviewSeverityDistribution
from raki.metrics.operational.verify_rate import FirstPassSuccessRate
from raki.metrics.protocol import MetricConfig
from raki.model import (
    EvalDataset,
    EvalSample,
    PhaseResult,
    ReviewFinding,
    SessionMeta,
)
from raki.model.report import SampleResult


# --- FirstPassSuccessRate ---


def test_first_pass_success_rate_all_pass():
    """All sessions with rework_cycles=0 should score 1.0."""
    dataset = make_dataset(
        make_sample("1", rework_cycles=0),
        make_sample("2", rework_cycles=0),
    )
    result = FirstPassSuccessRate().compute(dataset, MetricConfig())
    assert result.score == 1.0
    assert result.details["passed"] == 2
    assert result.details["total"] == 2


def test_first_pass_success_rate_all_rework():
    """All sessions with rework_cycles > 0 should score 0.0."""
    dataset = make_dataset(
        make_sample("1", rework_cycles=1),
        make_sample("2", rework_cycles=2),
    )
    result = FirstPassSuccessRate().compute(dataset, MetricConfig())
    assert result.score == 0.0
    assert result.details["passed"] == 0
    assert result.details["total"] == 2


def test_first_pass_success_rate_mixed():
    """Mix of rework and no-rework sessions gives correct ratio."""
    dataset = make_dataset(
        make_sample("1", rework_cycles=0),
        make_sample("2", rework_cycles=1),
        make_sample("3", rework_cycles=0),
    )
    result = FirstPassSuccessRate().compute(dataset, MetricConfig())
    assert abs(result.score - 2 / 3) < 0.01
    assert result.sample_scores["1"] == 1.0
    assert result.sample_scores["2"] == 0.0
    assert result.sample_scores["3"] == 1.0


def test_first_pass_success_rate_empty_dataset():
    """Empty dataset should return score=None (no applicable data)."""
    dataset = EvalDataset(samples=[])
    result = FirstPassSuccessRate().compute(dataset, MetricConfig())
    assert result.score is None
    assert result.details["passed"] == 0
    assert result.details["total"] == 0


def test_first_pass_success_rate_properties():
    """Check all Protocol-required class attributes."""
    metric = FirstPassSuccessRate()
    assert metric.name == "first_pass_success_rate"
    assert metric.requires_ground_truth is False
    assert metric.requires_llm is False
    assert metric.higher_is_better is True
    assert metric.display_format == "percent"
    assert metric.display_name == "First-pass success rate"


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
    metrics = [FirstPassSuccessRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    report = engine.run(dataset, skip_llm=True)
    # Both operational metrics should be included (neither requires LLM)
    assert "first_pass_success_rate" in report.aggregate_scores
    assert "rework_cycles" in report.aggregate_scores


def test_engine_skips_ground_truth_metrics():
    """MetricsEngine.run() with skip_ground_truth=True should skip metrics requiring ground truth."""
    metrics = [FirstPassSuccessRate(), CostEfficiency()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    report = engine.run(dataset, skip_ground_truth=True)
    # Both operational metrics should be included (neither requires ground truth)
    assert "first_pass_success_rate" in report.aggregate_scores
    assert "cost_efficiency" in report.aggregate_scores


def test_engine_run_single():
    """MetricsEngine.run_single() runs only the named metric."""
    metrics = [FirstPassSuccessRate(), ReworkCycles()]
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

    metrics = [FirstPassSuccessRate(), ReworkCycles(), CostEfficiency()]
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

    metrics = [FirstPassSuccessRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    session_ids = {sr.sample.session.session_id for sr in report.sample_results}
    assert session_ids == {"session-a", "session-b"}


def test_engine_sample_results_contain_metric_scores():
    """Each SampleResult should contain per-metric scores for that session."""
    sample_a = make_sample("session-a", rework_cycles=0, cost=10.0)
    sample_b = make_sample("session-b", rework_cycles=2, cost=20.0)
    dataset = make_dataset(sample_a, sample_b)

    metrics = [FirstPassSuccessRate(), ReworkCycles(), CostEfficiency()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset)

    # Find sample_result for session-b
    result_b = next(
        sr for sr in report.sample_results if sr.sample.session.session_id == "session-b"
    )
    score_names = {metric_score.name for metric_score in result_b.scores}
    assert score_names == {"first_pass_success_rate", "rework_cycles", "cost_efficiency"}

    # Verify that sample-level scores match expected values
    rework_score = next(ms for ms in result_b.scores if ms.name == "rework_cycles")
    assert rework_score.score == 2.0

    cost_score = next(ms for ms in result_b.scores if ms.name == "cost_efficiency")
    assert cost_score.score == 20.0


def test_engine_sample_results_with_skipped_metrics():
    """SampleResult should only contain scores for metrics that were not skipped."""
    dataset = make_dataset(make_sample("session-a"))

    metrics = [FirstPassSuccessRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    report = engine.run(dataset, skip_llm=True)

    # Neither metric requires LLM, so both should appear
    assert len(report.sample_results) == 1
    score_names = {ms.name for ms in report.sample_results[0].scores}
    assert score_names == {"first_pass_success_rate", "rework_cycles"}


def test_engine_sample_results_empty_dataset():
    """engine.run() with an empty dataset should produce empty sample_results."""
    dataset = EvalDataset(samples=[])
    metrics = [FirstPassSuccessRate(), ReworkCycles()]
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


# --- KnowledgeRetrievalMissRate (removed) ---
# Old KnowledgeRetrievalMissRate tests removed -- metric decomposed into
# SelfCorrectionRate (operational), KnowledgeGapRate and KnowledgeMissRate
# (knowledge tier). See tests/test_metrics_knowledge.py for knowledge tests.


# --- PhaseExecutionTimeMetric ---


class TestPhaseExecutionTime:
    def test_computes_total_duration_per_session(self):
        """Two phases with 5000ms each = 10.0s total per session."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        sample = make_sample("s1", duration_ms=5000)
        dataset = make_dataset(sample)
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        # 2 phases * 5000ms = 10000ms = 10.0s
        assert result.score == pytest.approx(10.0)

    def test_skips_none_durations(self):
        """Sessions with all None durations should be excluded from the aggregate."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        sample = make_sample("s1", duration_ms=None)
        dataset = make_dataset(sample)
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        assert result.score == 0.0

    def test_per_session_scores(self):
        """Per-session scores should reflect total phase time in seconds."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        sample_a = make_sample("s1", duration_ms=3000)
        sample_b = make_sample("s2", duration_ms=6000)
        dataset = make_dataset(sample_a, sample_b)
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        # s1: 2 phases * 3000ms = 6.0s
        assert result.sample_scores["s1"] == pytest.approx(6.0)
        # s2: 2 phases * 6000ms = 12.0s
        assert result.sample_scores["s2"] == pytest.approx(12.0)

    def test_aggregate_is_mean_of_session_totals(self):
        """Aggregate score should be the mean of per-session totals."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        sample_a = make_sample("s1", duration_ms=3000)
        sample_b = make_sample("s2", duration_ms=6000)
        dataset = make_dataset(sample_a, sample_b)
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        # mean of 6.0s and 12.0s = 9.0s
        assert result.score == pytest.approx(9.0)

    def test_details_include_percentiles(self):
        """Details dict should include p50, p95, min, max keys."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        samples = [make_sample(f"s{idx}", duration_ms=idx * 1000) for idx in range(1, 6)]
        dataset = make_dataset(*samples)
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        assert "p50" in result.details
        assert "p95" in result.details
        assert "min" in result.details
        assert "max" in result.details

    def test_details_min_max_values(self):
        """Min and max should match the smallest and largest session totals."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        sample_a = make_sample("s1", duration_ms=2000)  # 4.0s total
        sample_b = make_sample("s2", duration_ms=8000)  # 16.0s total
        dataset = make_dataset(sample_a, sample_b)
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        assert result.details["min"] == pytest.approx(4.0)
        assert result.details["max"] == pytest.approx(16.0)

    def test_empty_dataset(self):
        """Empty dataset should produce score 0.0."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        dataset = EvalDataset(samples=[])
        metric = PhaseExecutionTimeMetric()
        result = metric.compute(dataset, MetricConfig())
        assert result.score == 0.0

    def test_metric_properties(self):
        """Verify Protocol-required class attributes."""
        from raki.metrics.operational.latency import PhaseExecutionTimeMetric

        metric = PhaseExecutionTimeMetric()
        assert metric.name == "phase_execution_time"
        assert metric.higher_is_better is False
        assert metric.display_format == "duration"
        assert metric.requires_ground_truth is False
        assert metric.requires_llm is False
        assert metric.display_name == "Phase execution time"


# --- TokenEfficiencyMetric ---


class TestTokenEfficiency:
    def test_computes_tokens_per_phase(self):
        """Each phase gets tokens_in=1000 + tokens_out=500 = 1500 per phase."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample = make_sample("s1", tokens_in=1000, tokens_out=500)
        dataset = make_dataset(sample)
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        # 2 phases, each with 1000+500=1500 tokens -> mean per phase = 1500.0
        assert result.score == pytest.approx(1500.0)

    def test_skips_phases_with_both_none_tokens(self):
        """Sessions with all None tokens should be excluded from the aggregate."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample = make_sample("s1", tokens_in=None, tokens_out=None)
        dataset = make_dataset(sample)
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        assert result.score == 0.0

    def test_partial_none_tokens_treated_as_zero(self):
        """Phases with only tokens_in=None but tokens_out set should count tokens_out."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample = make_sample("s1", tokens_in=None, tokens_out=500)
        dataset = make_dataset(sample)
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        # Each phase: (0 + 500) = 500 tokens -> mean per phase = 500.0
        assert result.score == pytest.approx(500.0)

    def test_per_session_scores(self):
        """Per-session scores should reflect average tokens per phase."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample_a = make_sample("s1", tokens_in=500, tokens_out=200)
        sample_b = make_sample("s2", tokens_in=2000, tokens_out=1000)
        dataset = make_dataset(sample_a, sample_b)
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        # s1: each phase = 700, mean = 700.0
        assert result.sample_scores["s1"] == pytest.approx(700.0)
        # s2: each phase = 3000, mean = 3000.0
        assert result.sample_scores["s2"] == pytest.approx(3000.0)

    def test_aggregate_is_mean_of_session_averages(self):
        """Aggregate score should be the mean of per-session averages."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample_a = make_sample("s1", tokens_in=500, tokens_out=200)
        sample_b = make_sample("s2", tokens_in=2000, tokens_out=1000)
        dataset = make_dataset(sample_a, sample_b)
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        # mean of 700.0 and 3000.0 = 1850.0
        assert result.score == pytest.approx(1850.0)

    def test_empty_dataset(self):
        """Empty dataset should produce score 0.0."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        dataset = EvalDataset(samples=[])
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        assert result.score == 0.0

    def test_details_include_session_count(self):
        """Details dict should include sessions_with_tokens count."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample_a = make_sample("s1", tokens_in=500, tokens_out=200)
        sample_b = make_sample("s2", tokens_in=None, tokens_out=None)
        dataset = make_dataset(sample_a, sample_b)
        metric = TokenEfficiencyMetric()
        result = metric.compute(dataset, MetricConfig())
        assert result.details["sessions_with_tokens"] == 1

    def test_metric_properties(self):
        """Verify Protocol-required class attributes."""
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        metric = TokenEfficiencyMetric()
        assert metric.name == "token_efficiency"
        assert metric.higher_is_better is False
        assert metric.display_format == "count"
        assert metric.requires_ground_truth is False
        assert metric.requires_llm is False
        assert metric.display_name == "Tokens / phase"


# --- SelfCorrectionRate ---


class TestSelfCorrectionRate:
    def test_resolved_findings_scores_one(self):
        """When rework resolves all findings (final verify passes), score is 1.0."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        meta = SessionMeta(
            session_id="rework-resolved",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=4,
            rework_cycles=1,
        )
        findings = [
            ReviewFinding(reviewer="ai-review", severity="critical", issue="Missing null check"),
            ReviewFinding(reviewer="ai-review", severity="major", issue="SQL injection risk"),
        ]
        phases = [
            PhaseResult(name="implement", generation=1, status="completed", output="done"),
            PhaseResult(name="review", generation=1, status="completed", output="findings"),
            PhaseResult(name="implement", generation=2, status="completed", output="fixed"),
            PhaseResult(name="verify", generation=2, status="completed", output="PASS"),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=findings, events=[])
        dataset = make_dataset(sample)
        result = SelfCorrectionRate().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.details["total_rework_findings"] == 2
        assert result.details["resolved_findings"] == 2

    def test_zero_rework_returns_na(self):
        """No rework findings means N/A (score=None)."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        dataset = make_dataset(
            make_sample("1", rework_cycles=0),
            make_sample("2", rework_cycles=0),
        )
        result = SelfCorrectionRate().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["total_rework_findings"] == 0

    def test_unresolved_findings_scores_zero(self):
        """When rework does not resolve findings (final verify fails), score is 0.0."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        meta = SessionMeta(
            session_id="rework-unresolved",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=4,
            rework_cycles=1,
        )
        findings = [
            ReviewFinding(reviewer="ai-review", severity="critical", issue="Missing null check"),
        ]
        phases = [
            PhaseResult(name="implement", generation=1, status="completed", output="done"),
            PhaseResult(name="review", generation=1, status="completed", output="findings"),
            PhaseResult(name="implement", generation=2, status="completed", output="attempted fix"),
            PhaseResult(name="verify", generation=2, status="failed", output="FAIL"),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=findings, events=[])
        dataset = make_dataset(sample)
        result = SelfCorrectionRate().compute(dataset, MetricConfig())
        assert result.score == 0.0
        assert result.details["total_rework_findings"] == 1
        assert result.details["resolved_findings"] == 0

    def test_mixed_sessions(self):
        """Mix of resolved and unresolved rework sessions computes correct ratio."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        # Session 1: rework with 2 findings, resolved (verify passes)
        meta_resolved = SessionMeta(
            session_id="resolved",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=4,
            rework_cycles=1,
        )
        findings_resolved = [
            ReviewFinding(reviewer="ai", severity="critical", issue="bug A"),
            ReviewFinding(reviewer="ai", severity="major", issue="bug B"),
        ]
        phases_resolved = [
            PhaseResult(name="implement", generation=1, status="completed", output="done"),
            PhaseResult(name="review", generation=1, status="completed", output="findings"),
            PhaseResult(name="implement", generation=2, status="completed", output="fixed"),
            PhaseResult(name="verify", generation=2, status="completed", output="PASS"),
        ]
        sample_resolved = EvalSample(
            session=meta_resolved,
            phases=phases_resolved,
            findings=findings_resolved,
            events=[],
        )

        # Session 2: rework with 1 finding, not resolved (verify fails)
        meta_unresolved = SessionMeta(
            session_id="unresolved",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=4,
            rework_cycles=1,
        )
        findings_unresolved = [
            ReviewFinding(reviewer="ai", severity="major", issue="bug C"),
        ]
        phases_unresolved = [
            PhaseResult(name="implement", generation=1, status="completed", output="done"),
            PhaseResult(name="review", generation=1, status="completed", output="findings"),
            PhaseResult(name="implement", generation=2, status="completed", output="tried"),
            PhaseResult(name="verify", generation=2, status="failed", output="FAIL"),
        ]
        sample_unresolved = EvalSample(
            session=meta_unresolved,
            phases=phases_unresolved,
            findings=findings_unresolved,
            events=[],
        )

        dataset = make_dataset(sample_resolved, sample_unresolved)
        result = SelfCorrectionRate().compute(dataset, MetricConfig())
        # 2 resolved out of 3 total
        assert result.score == pytest.approx(2 / 3)
        assert result.details["total_rework_findings"] == 3
        assert result.details["resolved_findings"] == 2

    def test_no_verify_phase_counts_as_unresolved(self):
        """Rework session without a verify phase counts findings as unresolved."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        meta = SessionMeta(
            session_id="no-verify",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        findings = [
            ReviewFinding(reviewer="ai", severity="major", issue="issue"),
        ]
        phases = [
            PhaseResult(name="implement", generation=1, status="completed", output="done"),
            PhaseResult(name="review", generation=1, status="completed", output="findings"),
            PhaseResult(name="implement", generation=2, status="completed", output="fix attempt"),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=findings, events=[])
        dataset = make_dataset(sample)
        result = SelfCorrectionRate().compute(dataset, MetricConfig())
        assert result.score == 0.0
        assert result.details["total_rework_findings"] == 1
        assert result.details["resolved_findings"] == 0

    def test_minor_only_findings_returns_na(self):
        """When all findings are minor severity, they are filtered out and result is N/A."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        meta = SessionMeta(
            session_id="minor-only",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=4,
            rework_cycles=1,
        )
        findings = [
            ReviewFinding(reviewer="ai-review", severity="minor", issue="Style nit"),
            ReviewFinding(reviewer="ai-review", severity="minor", issue="Naming convention"),
        ]
        phases = [
            PhaseResult(name="implement", generation=1, status="completed", output="done"),
            PhaseResult(name="review", generation=1, status="completed", output="findings"),
            PhaseResult(name="implement", generation=2, status="completed", output="fixed"),
            PhaseResult(name="verify", generation=2, status="completed", output="PASS"),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=findings, events=[])
        dataset = make_dataset(sample)
        result = SelfCorrectionRate().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["total_rework_findings"] == 0

    def test_properties(self):
        """Check metric protocol attributes."""
        from raki.metrics.operational.self_correction import SelfCorrectionRate

        metric = SelfCorrectionRate()
        assert metric.name == "self_correction_rate"
        assert metric.requires_llm is False
        assert metric.requires_ground_truth is False
        assert metric.higher_is_better is True
        assert metric.display_format == "percent"
        assert metric.display_name == "Self-correction rate"
