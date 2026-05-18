"""Tests for reaggregate_scores() utility function.

6 unit tests covering: known scores, None handling, all-None edge case,
missing metrics, empty input, single session.
"""

import pytest

from conftest import make_sample
from raki.metrics.reaggregate import reaggregate_scores
from raki.model.report import MetricResult, SampleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sample_result(
    session_id: str,
    scores: list[tuple[str, float | None]],
) -> SampleResult:
    """Build a SampleResult with the given per-metric scores."""
    sample = make_sample(session_id)
    metric_results = [MetricResult(name=name, score=score) for name, score in scores]
    return SampleResult(sample=sample, scores=metric_results)


# ---------------------------------------------------------------------------
# Unit tests (Task 1)
# ---------------------------------------------------------------------------


def test_known_scores_computes_correct_mean():
    """Mean of known scores must be computed correctly across sessions."""
    sample_results = [
        make_sample_result("s1", [("rework_cycles", 0.0), ("cost_efficiency", 10.0)]),
        make_sample_result("s2", [("rework_cycles", 2.0), ("cost_efficiency", 20.0)]),
        make_sample_result("s3", [("rework_cycles", 1.0), ("cost_efficiency", 15.0)]),
    ]
    result = reaggregate_scores(sample_results)

    assert result["rework_cycles"] == pytest.approx(1.0)  # (0+2+1)/3
    assert result["cost_efficiency"] == pytest.approx(15.0)  # (10+20+15)/3


def test_none_scores_are_skipped_from_mean():
    """None scores for a metric are excluded from the mean calculation."""
    sample_results = [
        make_sample_result("s1", [("first_pass_success_rate", 1.0)]),
        make_sample_result("s2", [("first_pass_success_rate", None)]),
        make_sample_result("s3", [("first_pass_success_rate", 0.0)]),
    ]
    result = reaggregate_scores(sample_results)

    # Only s1 (1.0) and s3 (0.0) contribute; s2 is skipped
    assert result["first_pass_success_rate"] == pytest.approx(0.5)


def test_all_none_scores_returns_none():
    """When every score for a metric is None, reaggregate_scores returns None."""
    sample_results = [
        make_sample_result("s1", [("self_correction_rate", None)]),
        make_sample_result("s2", [("self_correction_rate", None)]),
    ]
    result = reaggregate_scores(sample_results)

    assert result["self_correction_rate"] is None


def test_missing_metric_uses_only_present_samples():
    """A metric absent from some samples is averaged over those that do have it."""
    sample_results = [
        make_sample_result("s1", [("rework_cycles", 2.0)]),
        make_sample_result("s2", [("rework_cycles", 4.0), ("cost_efficiency", 20.0)]),
        make_sample_result("s3", [("cost_efficiency", 10.0)]),
    ]
    result = reaggregate_scores(sample_results)

    # rework_cycles only in s1 and s2 -> (2+4)/2 = 3.0
    assert result["rework_cycles"] == pytest.approx(3.0)
    # cost_efficiency only in s2 and s3 -> (20+10)/2 = 15.0
    assert result["cost_efficiency"] == pytest.approx(15.0)


def test_empty_sample_results_returns_empty_dict():
    """An empty list of SampleResults produces an empty dict."""
    result = reaggregate_scores([])
    assert result == {}


def test_single_session_returns_its_scores():
    """A single SampleResult with non-None scores returns those scores unchanged."""
    sample_results = [
        make_sample_result(
            "s1",
            [("rework_cycles", 3.0), ("cost_efficiency", 42.0)],
        )
    ]
    result = reaggregate_scores(sample_results)

    assert result["rework_cycles"] == pytest.approx(3.0)
    assert result["cost_efficiency"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Round-trip integration test (Task 2)
# ---------------------------------------------------------------------------


def test_round_trip_reaggregated_matches_engine_aggregate():
    """reaggregate_scores(report.sample_results) must match report.aggregate_scores
    for all metrics that support per-sample scoring.

    Metrics explicitly skipped:
    - review_severity_distribution: aggregate-only (no sample_scores), absent from
      sample_results.scores and therefore from reaggregated output.
    - self_correction_rate: ratio-of-sums (resolved/total across all sessions) vs
      mean-of-ratios (mean of per-session 0.0/1.0), which differ when sessions have
      unequal finding counts.
    """
    from conftest import make_dataset

    from raki.metrics.engine import MetricsEngine
    from raki.metrics.operational import ALL_OPERATIONAL

    samples = [
        make_sample(
            "s1", rework_cycles=0, cost=10.0, duration_ms=3000, tokens_in=500, tokens_out=200
        ),
        make_sample(
            "s2", rework_cycles=2, cost=20.0, duration_ms=6000, tokens_in=1000, tokens_out=400
        ),
        make_sample(
            "s3", rework_cycles=0, cost=15.0, duration_ms=4500, tokens_in=750, tokens_out=300
        ),
    ]
    dataset = make_dataset(*samples)
    engine = MetricsEngine(metrics=ALL_OPERATIONAL)
    report = engine.run(dataset, skip_judge=True)

    reaggregated = reaggregate_scores(report.sample_results)

    # Metrics that support per-sample scores and round-trip cleanly
    roundtrip_metrics = [
        "first_pass_success_rate",
        "rework_cycles",
        "cost_efficiency",
        "phase_execution_time",
        "token_efficiency",
        "triage_calibration",
        "file_prediction_accuracy",
    ]

    for metric_name in roundtrip_metrics:
        if metric_name not in report.aggregate_scores:
            continue  # metric was skipped (e.g., requires ground truth)
        engine_score = report.aggregate_scores[metric_name]
        reagg_score = reaggregated.get(metric_name)

        if engine_score is None:
            assert reagg_score is None, f"{metric_name}: engine=None but reaggregated={reagg_score}"
        else:
            assert reagg_score is not None, (
                f"{metric_name}: engine={engine_score} but reaggregated=None"
            )
            assert reagg_score == pytest.approx(engine_score, rel=1e-6), (
                f"{metric_name}: engine={engine_score} != reaggregated={reagg_score}"
            )

    # Aggregate-only metrics must be absent from reaggregated output
    assert "review_severity_distribution" not in reaggregated, (
        "review_severity_distribution is aggregate-only; must not appear in reaggregated output"
    )
