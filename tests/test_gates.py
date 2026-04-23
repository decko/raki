"""Tests for quality gates: threshold parsing/evaluation and regression detection."""

import pytest

from raki.gates.thresholds import (
    Threshold,
    ThresholdResult,
    evaluate_all,
    evaluate_threshold,
    format_threshold_results,
    parse_threshold,
)
from raki.gates.regression import (
    compute_exit_code,
    detect_regressions,
)


class TestThresholdParser:
    """Tests for parse_threshold() — all operators, invalid syntax, edge cases."""

    def test_parse_greater_than(self):
        result = parse_threshold("faithfulness>0.85")
        assert result == Threshold(metric="faithfulness", operator=">", value=0.85)

    def test_parse_less_than(self):
        result = parse_threshold("rework_cycles<2.0")
        assert result == Threshold(metric="rework_cycles", operator="<", value=2.0)

    def test_parse_greater_equal(self):
        result = parse_threshold("context_precision>=0.90")
        assert result == Threshold(metric="context_precision", operator=">=", value=0.90)

    def test_parse_less_equal(self):
        result = parse_threshold("cost_efficiency<=15.0")
        assert result == Threshold(metric="cost_efficiency", operator="<=", value=15.0)

    def test_parse_integer_value(self):
        result = parse_threshold("rework_cycles<3")
        assert result == Threshold(metric="rework_cycles", operator="<", value=3.0)

    def test_parse_zero_value(self):
        result = parse_threshold("rework_cycles>=0")
        assert result == Threshold(metric="rework_cycles", operator=">=", value=0.0)

    def test_parse_invalid_no_operator(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            parse_threshold("faithfulness0.85")

    def test_parse_invalid_empty_string(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            parse_threshold("")

    def test_parse_invalid_missing_value(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            parse_threshold("faithfulness>")

    def test_parse_invalid_missing_metric(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            parse_threshold(">0.85")

    def test_parse_invalid_non_numeric_value(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            parse_threshold("faithfulness>abc")

    def test_parse_negative_value(self):
        result = parse_threshold("some_metric>-0.5")
        assert result == Threshold(metric="some_metric", operator=">", value=-0.5)

    def test_parse_metric_with_underscores(self):
        result = parse_threshold("first_pass_success_rate>0.8")
        assert result.metric == "first_pass_success_rate"

    def test_parse_whitespace_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid threshold"):
            parse_threshold("faithfulness > 0.85")


class TestThresholdEvaluator:
    """Tests for evaluate_threshold() — pass, fail, N/A skip, N/A required, unknown metric."""

    def test_pass_greater_than(self):
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores = {"faithfulness": 0.90}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is True
        assert result.skipped is False
        assert result.actual == 0.90

    def test_fail_greater_than(self):
        threshold = Threshold(metric="faithfulness", operator=">", value=0.90)
        scores = {"faithfulness": 0.85}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is False
        assert result.skipped is False

    def test_pass_less_than(self):
        threshold = Threshold(metric="rework_cycles", operator="<", value=2.0)
        scores = {"rework_cycles": 1.0}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is True

    def test_fail_less_than(self):
        threshold = Threshold(metric="rework_cycles", operator="<", value=1.0)
        scores = {"rework_cycles": 2.0}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is False

    def test_pass_greater_equal_at_boundary(self):
        threshold = Threshold(metric="faithfulness", operator=">=", value=0.85)
        scores = {"faithfulness": 0.85}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is True

    def test_pass_less_equal_at_boundary(self):
        threshold = Threshold(metric="rework_cycles", operator="<=", value=1.5)
        scores = {"rework_cycles": 1.5}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is True

    def test_unknown_metric_skipped_not_raised(self):
        """Unknown metric (not in scores, not required) is skipped gracefully."""
        threshold = Threshold(metric="nonexistent_metric", operator=">", value=0.5)
        scores: dict[str, float | None] = {"faithfulness": 0.90}
        result = evaluate_threshold(threshold, scores)
        assert result.skipped is True
        assert result.passed is True
        assert result.actual is None
        assert "not computed" in result.reason.lower()

    def test_none_score_skipped_by_default(self):
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores: dict[str, float | None] = {"faithfulness": None}
        result = evaluate_threshold(threshold, scores)
        assert result.skipped is True
        assert result.passed is True  # skipped counts as passed (not a violation)
        assert result.actual is None
        assert "N/A" in result.reason

    def test_none_score_fails_when_required(self):
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores: dict[str, float | None] = {"faithfulness": None}
        result = evaluate_threshold(threshold, scores, required_metrics={"faithfulness"})
        assert result.skipped is False
        assert result.passed is False
        assert "required" in result.reason.lower()

    def test_none_score_skipped_when_not_in_required_set(self):
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores: dict[str, float | None] = {"faithfulness": None}
        result = evaluate_threshold(threshold, scores, required_metrics={"context_precision"})
        assert result.skipped is True
        assert result.passed is True

    def test_missing_metric_fails_when_required(self):
        """Bug #137: --require-metric with a metric not in scores should fail gracefully."""
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores: dict[str, float | None] = {"rework_cycles": 1.5}
        result = evaluate_threshold(threshold, scores, required_metrics={"faithfulness"})
        assert result.passed is False
        assert result.skipped is False
        assert result.actual is None
        assert "not computed" in result.reason.lower()

    def test_missing_metric_skipped_when_not_required(self):
        """Bug #137: metric absent from scores and not required should skip, not crash."""
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores: dict[str, float | None] = {"rework_cycles": 1.5}
        result = evaluate_threshold(threshold, scores)
        assert result.skipped is True
        assert result.passed is True
        assert result.actual is None
        assert "not computed" in result.reason.lower()

    def test_missing_metric_skipped_when_required_set_does_not_include_it(self):
        """Bug #137: metric absent from scores but not in required set should skip."""
        threshold = Threshold(metric="faithfulness", operator=">", value=0.80)
        scores: dict[str, float | None] = {"rework_cycles": 1.5}
        result = evaluate_threshold(threshold, scores, required_metrics={"context_precision"})
        assert result.skipped is True
        assert result.passed is True


class TestEvaluateAll:
    """Tests for evaluate_all() — multiple thresholds, mixed results."""

    def test_all_pass(self):
        thresholds = [
            Threshold(metric="faithfulness", operator=">", value=0.80),
            Threshold(metric="rework_cycles", operator="<", value=2.0),
        ]
        scores: dict[str, float | None] = {
            "faithfulness": 0.95,
            "rework_cycles": 1.0,
        }
        results = evaluate_all(thresholds, scores)
        assert len(results) == 2
        assert all(result.passed for result in results)

    def test_mixed_pass_fail(self):
        thresholds = [
            Threshold(metric="faithfulness", operator=">", value=0.90),
            Threshold(metric="rework_cycles", operator="<", value=1.0),
        ]
        scores: dict[str, float | None] = {
            "faithfulness": 0.95,  # passes
            "rework_cycles": 2.0,  # fails
        }
        results = evaluate_all(thresholds, scores)
        assert results[0].passed is True
        assert results[1].passed is False

    def test_with_skipped(self):
        thresholds = [
            Threshold(metric="faithfulness", operator=">", value=0.80),
            Threshold(metric="context_precision", operator=">=", value=0.70),
        ]
        scores: dict[str, float | None] = {
            "faithfulness": 0.95,
            "context_precision": None,
        }
        results = evaluate_all(thresholds, scores)
        assert results[0].passed is True
        assert results[1].skipped is True

    def test_with_required_metrics(self):
        thresholds = [
            Threshold(metric="faithfulness", operator=">", value=0.80),
        ]
        scores: dict[str, float | None] = {
            "faithfulness": None,
        }
        results = evaluate_all(thresholds, scores, required_metrics={"faithfulness"})
        assert results[0].passed is False
        assert results[0].skipped is False

    def test_empty_thresholds(self):
        results = evaluate_all([], {"faithfulness": 0.90})
        assert results == []

    def test_missing_metric_required_no_traceback(self):
        """Bug #137: evaluate_all should not raise when required metric is absent from scores."""
        thresholds = [
            Threshold(metric="faithfulness", operator=">", value=0.80),
        ]
        scores: dict[str, float | None] = {"rework_cycles": 1.5}
        results = evaluate_all(thresholds, scores, required_metrics={"faithfulness"})
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].skipped is False
        assert "not computed" in results[0].reason.lower()


class TestFormatResults:
    """Tests for format_threshold_results() — formatting output."""

    def test_pass_format(self):
        results = [
            ThresholdResult(
                threshold=Threshold(metric="faithfulness", operator=">", value=0.80),
                actual=0.95,
                passed=True,
            ),
        ]
        output = format_threshold_results(results)
        assert "PASS" in output
        assert "faithfulness" in output

    def test_fail_format(self):
        results = [
            ThresholdResult(
                threshold=Threshold(metric="faithfulness", operator=">", value=0.90),
                actual=0.85,
                passed=False,
            ),
        ]
        output = format_threshold_results(results)
        assert "FAIL" in output
        assert "faithfulness" in output

    def test_skip_format(self):
        results = [
            ThresholdResult(
                threshold=Threshold(metric="faithfulness", operator=">", value=0.80),
                actual=None,
                passed=True,
                skipped=True,
                reason="Metric is N/A",
            ),
        ]
        output = format_threshold_results(results)
        assert "SKIP" in output
        assert "faithfulness" in output

    def test_actual_value_is_rounded_not_raw_float(self):
        """Bug #141: format_threshold_results should round actual, not show raw float."""
        results = [
            ThresholdResult(
                threshold=Threshold(metric="faithfulness", operator=">", value=0.80),
                actual=0.8333333333333334,
                passed=True,
            ),
        ]
        output = format_threshold_results(results)
        assert "0.8333333333333334" not in output
        assert "0.8333" in output

    def test_actual_value_rounded_on_fail(self):
        """Bug #141: FAIL line should also display a rounded actual value."""
        results = [
            ThresholdResult(
                threshold=Threshold(metric="rework_cycles", operator="<", value=1.0),
                actual=2.6666666666666665,
                passed=False,
            ),
        ]
        output = format_threshold_results(results)
        assert "2.6666666666666665" not in output
        assert "2.6667" in output

    def test_multiple_results_formatted(self):
        results = [
            ThresholdResult(
                threshold=Threshold(metric="faithfulness", operator=">", value=0.80),
                actual=0.95,
                passed=True,
            ),
            ThresholdResult(
                threshold=Threshold(metric="rework_cycles", operator="<", value=1.0),
                actual=2.0,
                passed=False,
            ),
        ]
        output = format_threshold_results(results)
        assert "PASS" in output
        assert "FAIL" in output


class TestLowerIsBetterThreshold:
    """Tests for threshold evaluation with lower_is_better metrics."""

    def test_lower_is_better_threshold_violation(self):
        """Threshold 'knowledge_gap_rate<0.20' should fail when score is 0.30."""
        threshold = parse_threshold("knowledge_gap_rate<0.20")
        scores: dict[str, float | None] = {"knowledge_gap_rate": 0.30}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is False

    def test_lower_is_better_threshold_passes(self):
        """Threshold 'knowledge_gap_rate<0.20' should pass when score is 0.10."""
        threshold = parse_threshold("knowledge_gap_rate<0.20")
        scores: dict[str, float | None] = {"knowledge_gap_rate": 0.10}
        result = evaluate_threshold(threshold, scores)
        assert result.passed is True


class TestRegressionDetection:
    """Tests for detect_regressions() — regression, improvement, noise margin, None scores."""

    def test_regression_higher_is_better(self):
        baseline = {"faithfulness": 0.90}
        current = {"faithfulness": 0.80}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 1
        assert results[0].regressed is True

    def test_improvement_higher_is_better(self):
        baseline = {"faithfulness": 0.80}
        current = {"faithfulness": 0.90}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 1
        assert results[0].regressed is False

    def test_regression_lower_is_better(self):
        baseline = {"rework_cycles": 1.0}
        current = {"rework_cycles": 2.5}
        directions = {"rework_cycles": "lower_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 1
        assert results[0].regressed is True

    def test_improvement_lower_is_better(self):
        baseline = {"rework_cycles": 2.5}
        current = {"rework_cycles": 1.0}
        directions = {"rework_cycles": "lower_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 1
        assert results[0].regressed is False

    def test_noise_margin_suppresses_small_regression(self):
        baseline = {"faithfulness": 0.90}
        current = {"faithfulness": 0.89}  # delta of 0.01 < noise_margin of 0.02
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions, noise_margin=0.02)
        assert len(results) == 1
        assert results[0].regressed is False

    def test_noise_margin_allows_real_regression(self):
        baseline = {"faithfulness": 0.90}
        current = {"faithfulness": 0.85}  # delta of 0.05 > noise_margin of 0.02
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions, noise_margin=0.02)
        assert len(results) == 1
        assert results[0].regressed is True

    def test_none_baseline_skipped(self):
        baseline: dict[str, float | None] = {"faithfulness": None}
        current: dict[str, float | None] = {"faithfulness": 0.90}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 0

    def test_none_current_skipped(self):
        baseline: dict[str, float | None] = {"faithfulness": 0.90}
        current: dict[str, float | None] = {"faithfulness": None}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 0

    def test_metric_only_in_baseline_skipped(self):
        baseline = {"faithfulness": 0.90}
        current: dict[str, float | None] = {}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 0

    def test_metric_only_in_current_skipped(self):
        baseline: dict[str, float | None] = {}
        current = {"faithfulness": 0.90}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 0

    def test_multiple_metrics_mixed(self):
        baseline = {"faithfulness": 0.90, "rework_cycles": 1.0}
        current = {"faithfulness": 0.80, "rework_cycles": 0.5}
        directions = {
            "faithfulness": "higher_is_better",
            "rework_cycles": "lower_is_better",
        }
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 2
        faith_result = next(result for result in results if result.metric == "faithfulness")
        rework_result = next(result for result in results if result.metric == "rework_cycles")
        assert faith_result.regressed is True
        assert rework_result.regressed is False

    def test_custom_noise_margin(self):
        baseline = {"faithfulness": 0.90}
        current = {"faithfulness": 0.85}
        directions = {"faithfulness": "higher_is_better"}
        # With noise_margin=0.10, a 0.05 drop is within the margin
        results = detect_regressions(baseline, current, directions, noise_margin=0.10)
        assert results[0].regressed is False

    def test_exact_same_score_no_regression(self):
        baseline = {"faithfulness": 0.90}
        current = {"faithfulness": 0.90}
        directions = {"faithfulness": "higher_is_better"}
        results = detect_regressions(baseline, current, directions)
        assert len(results) == 1
        assert results[0].regressed is False

    def test_regression_lower_is_better_knowledge_gap(self):
        """For lower_is_better metrics, an increase is a regression."""
        baseline: dict[str, float | None] = {"knowledge_gap_rate": 0.10}
        current: dict[str, float | None] = {"knowledge_gap_rate": 0.30}
        directions: dict[str, str] = {"knowledge_gap_rate": "lower_is_better"}
        regressions = detect_regressions(baseline, current, directions)
        assert len(regressions) == 1
        assert regressions[0].regressed is True


class TestExitCodes:
    """Tests for compute_exit_code() — all combinations."""

    def test_clear(self):
        assert compute_exit_code(threshold_violated=False, regression_detected=False) == 0

    def test_threshold_violation_only(self):
        assert compute_exit_code(threshold_violated=True, regression_detected=False) == 1

    def test_regression_only(self):
        assert compute_exit_code(threshold_violated=False, regression_detected=True) == 3

    def test_both(self):
        assert compute_exit_code(threshold_violated=True, regression_detected=True) == 4
