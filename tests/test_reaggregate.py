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
