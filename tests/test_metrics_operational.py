"""Tests for operational metrics: verify rate, rework cycles, severity, cost."""

from datetime import datetime, timezone

import pytest

from conftest import make_dataset, make_sample
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
    from raki.metrics.engine import MetricsEngine

    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    report = engine.run(dataset, skip_llm=True)
    # Both operational metrics should be included (neither requires LLM)
    assert "first_pass_verify_rate" in report.aggregate_scores
    assert "rework_cycles" in report.aggregate_scores


def test_engine_skips_ground_truth_metrics():
    """MetricsEngine.run() with skip_ground_truth=True should skip metrics requiring ground truth."""
    from raki.metrics.engine import MetricsEngine

    metrics = [FirstPassVerifyRate(), CostEfficiency()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    report = engine.run(dataset, skip_ground_truth=True)
    # Both operational metrics should be included (neither requires ground truth)
    assert "first_pass_verify_rate" in report.aggregate_scores
    assert "cost_efficiency" in report.aggregate_scores


def test_engine_run_single():
    """MetricsEngine.run_single() runs only the named metric."""
    from raki.metrics.engine import MetricsEngine

    metrics = [FirstPassVerifyRate(), ReworkCycles()]
    engine = MetricsEngine(metrics=metrics)
    dataset = make_dataset(make_sample("1"))
    result = engine.run_single("rework_cycles", dataset)
    assert result.name == "rework_cycles"

    with pytest.raises(ValueError, match="Unknown metric"):
        engine.run_single("nonexistent_metric", dataset)
