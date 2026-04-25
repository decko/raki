"""Tests for metric health checks (dead_metric, degenerate_metric)."""

from __future__ import annotations


from raki.metrics.health import run_health_checks
from raki.model.report import MetricResult, MetricWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(name: str, sample_scores: dict[str, float]) -> MetricResult:
    """Build a MetricResult with the given per-session scores."""
    return MetricResult(name=name, score=None, sample_scores=sample_scores)


def _find_warning(warnings: list[MetricWarning], check: str) -> MetricWarning | None:
    """Return the first warning matching *check*, or None."""
    for warning in warnings:
        if warning.check == check:
            return warning
    return None


# ---------------------------------------------------------------------------
# Skipping aggregate-only and empty datasets
# ---------------------------------------------------------------------------


class TestSkipConditions:
    def test_empty_sample_scores_returns_no_warnings(self) -> None:
        """Aggregate-only metrics (empty sample_scores) must be skipped entirely."""
        result = MetricResult(name="review_severity_distribution", score=0.9)
        warnings = run_health_checks(result, total_sessions=10)
        assert warnings == []

    def test_zero_total_sessions_returns_no_warnings(self) -> None:
        """When total_sessions is 0 we cannot compute rates — skip gracefully."""
        result = _result("some_metric", {"s1": 0.5})
        warnings = run_health_checks(result, total_sessions=0)
        assert warnings == []


# ---------------------------------------------------------------------------
# Dead metric check
# ---------------------------------------------------------------------------


class TestDeadMetricCheck:
    def test_metric_above_95_pct_na_fires_error(self) -> None:
        """Metric N/A for >95% of sessions triggers a dead_metric error."""
        # 1 session scored out of 100 → 99% N/A → dead
        result = _result("token_efficiency", {"s1": 0.5})
        warnings = run_health_checks(result, total_sessions=100)
        dead = _find_warning(warnings, "dead_metric")
        assert dead is not None
        assert dead.severity == "error"
        assert dead.metric_name == "token_efficiency"
        assert "dead_metric" == dead.check

    def test_metric_exactly_at_95_pct_na_does_not_fire(self) -> None:
        """Metric N/A for exactly 95% of sessions should NOT trigger dead_metric."""
        # 5 sessions scored out of 100 → exactly 95% N/A → boundary, should NOT fire
        sample_scores = {f"s{idx}": 0.5 for idx in range(5)}
        result = _result("token_efficiency", sample_scores)
        warnings = run_health_checks(result, total_sessions=100)
        dead = _find_warning(warnings, "dead_metric")
        assert dead is None

    def test_metric_below_95_pct_na_does_not_fire(self) -> None:
        """Metric N/A for <95% of sessions should NOT trigger dead_metric."""
        # 50 sessions scored out of 100 → 50% N/A → healthy
        sample_scores = {f"s{idx}": 0.5 for idx in range(50)}
        result = _result("rework_cycles", sample_scores)
        warnings = run_health_checks(result, total_sessions=100)
        dead = _find_warning(warnings, "dead_metric")
        assert dead is None

    def test_dead_metric_message_mentions_na_rate(self) -> None:
        """Dead metric warning message should mention the N/A rate."""
        # 1 session scored out of 21 → ~95.2% N/A → strictly > 95% threshold → fires
        result = _result("some_metric", {"s1": 0.5})
        warnings = run_health_checks(result, total_sessions=21)
        dead = _find_warning(warnings, "dead_metric")
        assert dead is not None
        assert "95%" in dead.message or "N/A" in dead.message

    def test_all_sessions_scored_no_dead_metric(self) -> None:
        """When all sessions are scored, no dead_metric warning is emitted."""
        sample_scores = {f"s{idx}": float(idx) / 10 for idx in range(10)}
        result = _result("first_pass_success_rate", sample_scores)
        warnings = run_health_checks(result, total_sessions=10)
        dead = _find_warning(warnings, "dead_metric")
        assert dead is None


# ---------------------------------------------------------------------------
# Degenerate metric check
# ---------------------------------------------------------------------------


class TestDegenerateMetricCheck:
    def test_constant_score_fires_warning(self) -> None:
        """Metric with a constant score across all sessions triggers degenerate_metric."""
        sample_scores = {f"s{idx}": 1.0 for idx in range(10)}
        result = _result("first_pass_success_rate", sample_scores)
        warnings = run_health_checks(result, total_sessions=10)
        degen = _find_warning(warnings, "degenerate_metric")
        assert degen is not None
        assert degen.severity == "warning"
        assert degen.metric_name == "first_pass_success_rate"

    def test_constant_zero_score_fires_warning(self) -> None:
        """A constant score of 0.0 (not just 1.0) should also trigger degenerate_metric."""
        sample_scores = {f"s{idx}": 0.0 for idx in range(5)}
        result = _result("some_metric", sample_scores)
        warnings = run_health_checks(result, total_sessions=5)
        degen = _find_warning(warnings, "degenerate_metric")
        assert degen is not None

    def test_two_distinct_scores_no_degenerate(self) -> None:
        """Two different scores → not degenerate."""
        result = _result("some_metric", {"s1": 0.0, "s2": 1.0})
        warnings = run_health_checks(result, total_sessions=2)
        degen = _find_warning(warnings, "degenerate_metric")
        assert degen is None

    def test_varied_scores_no_degenerate(self) -> None:
        """Many different scores → not degenerate."""
        sample_scores = {f"s{idx}": idx * 0.1 for idx in range(10)}
        result = _result("rework_cycles", sample_scores)
        warnings = run_health_checks(result, total_sessions=10)
        degen = _find_warning(warnings, "degenerate_metric")
        assert degen is None

    def test_single_session_scores_still_degenerate(self) -> None:
        """A single scored session is trivially constant — degenerate_metric fires."""
        result = _result("cost_efficiency", {"s1": 5.0})
        warnings = run_health_checks(result, total_sessions=100)
        # This should also trigger dead_metric (99% N/A), but degenerate should also fire.
        degen = _find_warning(warnings, "degenerate_metric")
        assert degen is not None

    def test_degenerate_message_mentions_constant_value(self) -> None:
        """Degenerate metric message should mention the constant score value."""
        sample_scores = {f"s{idx}": 0.75 for idx in range(5)}
        result = _result("some_metric", sample_scores)
        warnings = run_health_checks(result, total_sessions=5)
        degen = _find_warning(warnings, "degenerate_metric")
        assert degen is not None
        assert "0.75" in degen.message


# ---------------------------------------------------------------------------
# Combined checks
# ---------------------------------------------------------------------------


class TestCombinedChecks:
    def test_both_checks_can_fire_simultaneously(self) -> None:
        """A metric that is dead AND degenerate can emit two warnings."""
        # 1 session out of 100 → dead; constant 1.0 → degenerate
        result = _result("some_metric", {"s1": 1.0})
        warnings = run_health_checks(result, total_sessions=100)
        checks = {w.check for w in warnings}
        assert "dead_metric" in checks
        assert "degenerate_metric" in checks

    def test_healthy_metric_returns_empty_list(self) -> None:
        """A metric with varied scores across most sessions returns no warnings."""
        sample_scores = {f"s{idx}": (idx % 2) * 1.0 for idx in range(10)}
        result = _result("first_pass_success_rate", sample_scores)
        warnings = run_health_checks(result, total_sessions=10)
        assert warnings == []

    def test_return_type_is_list_of_metric_warning(self) -> None:
        """run_health_checks always returns a list (even empty)."""
        result = _result("rework_cycles", {"s1": 1.0, "s2": 2.0})
        warnings = run_health_checks(result, total_sessions=2)
        assert isinstance(warnings, list)
        for warning in warnings:
            assert isinstance(warning, MetricWarning)
