"""Tests for diff module — session matching, delta computation, transition grouping."""

from datetime import datetime, timezone
from io import StringIO

import pytest
from rich.console import Console

from raki.model.report import EvalReport, MetricResult, SampleResult
from raki.report.cli_summary import print_diff_summary
from raki.report.diff import (
    DiffReport,
    MatchResult,
    MetricDelta,
    SessionTransition,
    compute_deltas,
    compute_transitions,
    generate_diff_report,
    match_sessions,
)

from tests.conftest import make_sample


def _make_eval_report(
    run_id: str,
    aggregate_scores: dict[str, float],
    sample_results: list[SampleResult] | None = None,
) -> EvalReport:
    """Create a minimal EvalReport for diff testing."""
    return EvalReport(
        run_id=run_id,
        timestamp=datetime(2026, 4, 10, tzinfo=timezone.utc),
        aggregate_scores=aggregate_scores,
        sample_results=sample_results or [],
    )


def _make_sample_result(
    session_id: str,
    rework_cycles: int = 0,
    cost: float = 10.0,
    verify_status: str = "completed",
    scores: dict[str, float] | None = None,
) -> SampleResult:
    """Create a SampleResult for diff testing."""
    sample = make_sample(
        session_id=session_id,
        rework_cycles=rework_cycles,
        cost=cost,
        verify_status=verify_status,
    )
    metric_results = []
    if scores:
        for metric_name, score_value in scores.items():
            metric_results.append(
                MetricResult(
                    name=metric_name,
                    score=score_value,
                    sample_scores={session_id: score_value},
                )
            )
    return SampleResult(sample=sample, scores=metric_results)


class TestMatchSessions:
    def test_all_sessions_match(self):
        baseline = _make_eval_report(
            "base",
            {},
            [
                _make_sample_result("session-101"),
                _make_sample_result("session-102"),
            ],
        )
        compare = _make_eval_report(
            "comp",
            {},
            [
                _make_sample_result("session-101"),
                _make_sample_result("session-102"),
            ],
        )
        result = match_sessions(baseline, compare)
        assert isinstance(result, MatchResult)
        assert result.matched_ids == {"session-101", "session-102"}
        assert result.new_ids == set()
        assert result.dropped_ids == set()

    def test_new_sessions_in_compare(self):
        baseline = _make_eval_report("base", {}, [_make_sample_result("session-101")])
        compare = _make_eval_report(
            "comp",
            {},
            [
                _make_sample_result("session-101"),
                _make_sample_result("session-200"),
            ],
        )
        result = match_sessions(baseline, compare)
        assert result.matched_ids == {"session-101"}
        assert result.new_ids == {"session-200"}
        assert result.dropped_ids == set()

    def test_dropped_sessions_in_baseline(self):
        baseline = _make_eval_report(
            "base",
            {},
            [
                _make_sample_result("session-101"),
                _make_sample_result("session-102"),
            ],
        )
        compare = _make_eval_report("comp", {}, [_make_sample_result("session-101")])
        result = match_sessions(baseline, compare)
        assert result.matched_ids == {"session-101"}
        assert result.new_ids == set()
        assert result.dropped_ids == {"session-102"}

    def test_empty_reports_match_nothing(self):
        baseline = _make_eval_report("base", {})
        compare = _make_eval_report("comp", {})
        result = match_sessions(baseline, compare)
        assert result.matched_ids == set()
        assert result.new_ids == set()
        assert result.dropped_ids == set()

    def test_disjoint_sessions(self):
        baseline = _make_eval_report("base", {}, [_make_sample_result("session-101")])
        compare = _make_eval_report("comp", {}, [_make_sample_result("session-200")])
        result = match_sessions(baseline, compare)
        assert result.matched_ids == set()
        assert result.new_ids == {"session-200"}
        assert result.dropped_ids == {"session-101"}

    def test_counts_property(self):
        baseline = _make_eval_report(
            "base",
            {},
            [
                _make_sample_result("session-101"),
                _make_sample_result("session-102"),
                _make_sample_result("session-103"),
            ],
        )
        compare = _make_eval_report(
            "comp",
            {},
            [
                _make_sample_result("session-101"),
                _make_sample_result("session-200"),
            ],
        )
        result = match_sessions(baseline, compare)
        assert result.baseline_total == 3
        assert result.compare_total == 2
        assert len(result.matched_ids) == 1
        assert len(result.new_ids) == 1
        assert len(result.dropped_ids) == 2


class TestComputeDeltas:
    def test_higher_is_better_improvement(self):
        baseline_scores = {"first_pass_verify_rate": 0.78}
        compare_scores = {"first_pass_verify_rate": 0.91}
        deltas = compute_deltas(baseline_scores, compare_scores)
        assert len(deltas) == 1
        delta = deltas[0]
        assert delta.name == "first_pass_verify_rate"
        assert delta.baseline_value == pytest.approx(0.78)
        assert delta.compare_value == pytest.approx(0.91)
        assert delta.delta == pytest.approx(0.13)
        assert delta.direction == "improved"

    def test_higher_is_better_regression(self):
        baseline_scores = {"first_pass_verify_rate": 0.91}
        compare_scores = {"first_pass_verify_rate": 0.78}
        deltas = compute_deltas(baseline_scores, compare_scores)
        assert deltas[0].direction == "regressed"

    def test_lower_is_better_improvement(self):
        """Rework cycles: lower is better, so decrease = improvement."""
        baseline_scores = {"rework_cycles": 1.2}
        compare_scores = {"rework_cycles": 0.4}
        deltas = compute_deltas(baseline_scores, compare_scores)
        delta = deltas[0]
        assert delta.delta == pytest.approx(-0.8)
        assert delta.direction == "improved"

    def test_lower_is_better_regression(self):
        """Rework cycles: lower is better, so increase = regression."""
        baseline_scores = {"rework_cycles": 0.4}
        compare_scores = {"rework_cycles": 1.2}
        deltas = compute_deltas(baseline_scores, compare_scores)
        assert deltas[0].direction == "regressed"

    def test_flat_delta(self):
        baseline_scores = {"first_pass_verify_rate": 0.85}
        compare_scores = {"first_pass_verify_rate": 0.85}
        deltas = compute_deltas(baseline_scores, compare_scores)
        assert deltas[0].direction == "flat"
        assert deltas[0].delta == pytest.approx(0.0)

    def test_multiple_metrics(self):
        baseline_scores = {
            "first_pass_verify_rate": 0.78,
            "rework_cycles": 1.2,
            "cost_efficiency": 12.30,
        }
        compare_scores = {
            "first_pass_verify_rate": 0.91,
            "rework_cycles": 0.4,
            "cost_efficiency": 7.42,
        }
        deltas = compute_deltas(baseline_scores, compare_scores)
        assert len(deltas) == 3
        delta_names = {delta.name for delta in deltas}
        assert "first_pass_verify_rate" in delta_names
        assert "rework_cycles" in delta_names
        assert "cost_efficiency" in delta_names

    def test_metric_only_in_baseline_skipped(self):
        baseline_scores = {"first_pass_verify_rate": 0.78, "context_precision": 0.90}
        compare_scores = {"first_pass_verify_rate": 0.91}
        deltas = compute_deltas(baseline_scores, compare_scores)
        delta_names = {delta.name for delta in deltas}
        assert "first_pass_verify_rate" in delta_names
        # context_precision is only in baseline, should be skipped
        assert "context_precision" not in delta_names

    def test_metric_only_in_compare_skipped(self):
        baseline_scores = {"first_pass_verify_rate": 0.78}
        compare_scores = {"first_pass_verify_rate": 0.91, "context_precision": 0.90}
        deltas = compute_deltas(baseline_scores, compare_scores)
        delta_names = {delta.name for delta in deltas}
        assert "first_pass_verify_rate" in delta_names
        assert "context_precision" not in delta_names


class TestComputeTransitions:
    def test_pass_to_fail_is_regression(self):
        baseline_results = [_make_sample_result("session-101", verify_status="completed")]
        compare_results = [_make_sample_result("session-101", verify_status="failed")]
        matched_ids = {"session-101"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        assert len(transitions) == 1
        assert transitions[0].session_id == "session-101"
        assert transitions[0].old_verdict == "pass"
        assert transitions[0].new_verdict == "fail"
        assert transitions[0].transition_type == "regression"

    def test_fail_to_pass_is_improvement(self):
        baseline_results = [_make_sample_result("session-101", verify_status="failed")]
        compare_results = [_make_sample_result("session-101", verify_status="completed")]
        matched_ids = {"session-101"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        assert len(transitions) == 1
        assert transitions[0].transition_type == "improvement"

    def test_rework_to_pass_is_improvement(self):
        baseline_results = [_make_sample_result("session-101", rework_cycles=2)]
        compare_results = [_make_sample_result("session-101", rework_cycles=0)]
        matched_ids = {"session-101"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        assert len(transitions) == 1
        assert transitions[0].old_verdict == "rework"
        assert transitions[0].new_verdict == "pass"
        assert transitions[0].transition_type == "improvement"

    def test_pass_to_rework_is_regression(self):
        baseline_results = [_make_sample_result("session-101", rework_cycles=0)]
        compare_results = [_make_sample_result("session-101", rework_cycles=2)]
        matched_ids = {"session-101"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        assert len(transitions) == 1
        assert transitions[0].transition_type == "regression"

    def test_same_verdict_no_transition(self):
        baseline_results = [_make_sample_result("session-101")]
        compare_results = [_make_sample_result("session-101")]
        matched_ids = {"session-101"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        assert len(transitions) == 0

    def test_unmatched_sessions_excluded(self):
        baseline_results = [
            _make_sample_result("session-101"),
            _make_sample_result("session-200"),
        ]
        compare_results = [
            _make_sample_result("session-101"),
            _make_sample_result("session-300"),
        ]
        matched_ids = {"session-101"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        # session-101 is pass->pass (no change), others not matched
        assert len(transitions) == 0

    def test_multiple_transitions_sorted_regressions_first(self):
        baseline_results = [
            _make_sample_result("session-101", verify_status="completed"),
            _make_sample_result("session-102", verify_status="failed"),
        ]
        compare_results = [
            _make_sample_result("session-101", verify_status="failed"),
            _make_sample_result("session-102", verify_status="completed"),
        ]
        matched_ids = {"session-101", "session-102"}
        transitions = compute_transitions(baseline_results, compare_results, matched_ids)
        assert len(transitions) == 2
        # Regressions should come first
        assert transitions[0].transition_type == "regression"
        assert transitions[1].transition_type == "improvement"


class TestGenerateDiffReport:
    def test_produces_diff_report(self):
        baseline = _make_eval_report(
            "eval-abc123",
            {"first_pass_verify_rate": 0.78, "rework_cycles": 1.2},
            [_make_sample_result("session-101")],
        )
        compare = _make_eval_report(
            "eval-def456",
            {"first_pass_verify_rate": 0.91, "rework_cycles": 0.4},
            [_make_sample_result("session-101")],
        )
        diff = generate_diff_report(baseline, compare)
        assert isinstance(diff, DiffReport)
        assert diff.baseline_run_id == "eval-abc123"
        assert diff.compare_run_id == "eval-def456"
        assert isinstance(diff.match_result, MatchResult)
        assert len(diff.deltas) == 2

    def test_diff_report_with_transitions(self):
        baseline = _make_eval_report(
            "base",
            {"first_pass_verify_rate": 0.50},
            [_make_sample_result("session-101", verify_status="failed")],
        )
        compare = _make_eval_report(
            "comp",
            {"first_pass_verify_rate": 1.0},
            [_make_sample_result("session-101", verify_status="completed")],
        )
        diff = generate_diff_report(baseline, compare)
        assert len(diff.improvements) == 1
        assert len(diff.regressions) == 0

    def test_diff_report_empty_sample_results(self):
        """Diff should still work with aggregate-only reports (no sample_results)."""
        baseline = _make_eval_report("base", {"first_pass_verify_rate": 0.78})
        compare = _make_eval_report("comp", {"first_pass_verify_rate": 0.91})
        diff = generate_diff_report(baseline, compare)
        assert len(diff.deltas) == 1
        assert len(diff.improvements) == 0
        assert len(diff.regressions) == 0
        assert diff.has_session_data is False


class TestPrintDiffSummary:
    def _capture_diff_output(self, diff: DiffReport) -> str:
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, width=120)
        print_diff_summary(diff, console=test_console)
        return string_io.getvalue()

    def test_shows_comparing_header(self):
        diff = DiffReport(
            baseline_run_id="eval-abc123",
            compare_run_id="eval-def456",
            match_result=MatchResult(
                matched_ids={"session-101"},
                baseline_total=1,
                compare_total=1,
            ),
        )
        output = self._capture_diff_output(diff)
        assert "eval-abc123" in output
        assert "eval-def456" in output

    def test_shows_coverage_line(self):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(
                matched_ids={"s1", "s2"},
                new_ids={"s3"},
                dropped_ids={"s4"},
                baseline_total=3,
                compare_total=3,
            ),
        )
        output = self._capture_diff_output(diff)
        assert "2" in output  # matched count
        assert "new" in output.lower()
        assert "dropped" in output.lower()

    def test_shows_metric_deltas(self):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(matched_ids={"s1"}, baseline_total=1, compare_total=1),
            deltas=[
                MetricDelta(
                    name="first_pass_verify_rate",
                    baseline_value=0.78,
                    compare_value=0.91,
                    delta=0.13,
                    direction="improved",
                ),
            ],
        )
        output = self._capture_diff_output(diff)
        # Display format for verify rate is "percent", so values show as 78%, 91%
        assert "78%" in output
        assert "91%" in output
        assert "+13%" in output

    def test_shows_improvement_regression_counts(self):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(matched_ids={"s1", "s2"}, baseline_total=2, compare_total=2),
            improvements=[
                SessionTransition(
                    session_id="s1",
                    old_verdict="rework",
                    new_verdict="pass",
                    transition_type="improvement",
                ),
            ],
            regressions=[
                SessionTransition(
                    session_id="s2",
                    old_verdict="pass",
                    new_verdict="fail",
                    transition_type="regression",
                ),
            ],
        )
        output = self._capture_diff_output(diff)
        assert "improvement" in output.lower() or "Improvement" in output
        assert "regression" in output.lower() or "Regression" in output

    def test_warning_when_sessions_dropped_or_added(self):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(
                matched_ids={"s1"},
                new_ids={"s2", "s3"},
                dropped_ids={"s4"},
                baseline_total=2,
                compare_total=3,
            ),
        )
        output = self._capture_diff_output(diff)
        # Should contain a warning about dropped/new sessions
        assert "dropped" in output.lower()
        assert "new" in output.lower()

    def test_no_session_data_warning(self):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(),
            has_session_data=False,
        )
        output = self._capture_diff_output(diff)
        assert "include-sessions" in output.lower() or "unavailable" in output.lower()

    def test_direction_indicator_for_improved(self):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(matched_ids={"s1"}, baseline_total=1, compare_total=1),
            deltas=[
                MetricDelta(
                    name="first_pass_verify_rate",
                    baseline_value=0.78,
                    compare_value=0.91,
                    delta=0.13,
                    direction="improved",
                ),
            ],
        )
        output = self._capture_diff_output(diff)
        # Should have a green direction indicator
        assert "green" in output.lower() or "▲" in output


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("jinja2"),
    reason="jinja2 not installed",
)
class TestWriteDiffHtmlReport:
    def test_generates_html_file(self, tmp_path):
        diff = DiffReport(
            baseline_run_id="eval-base",
            compare_run_id="eval-comp",
            match_result=MatchResult(
                matched_ids={"s1"},
                baseline_total=1,
                compare_total=1,
            ),
            deltas=[
                MetricDelta(
                    name="first_pass_verify_rate",
                    baseline_value=0.78,
                    compare_value=0.91,
                    delta=0.13,
                    direction="improved",
                ),
            ],
        )
        from raki.report.html_report import write_diff_html_report

        html_path = tmp_path / "diff.html"
        write_diff_html_report(diff, html_path)
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content.lower()
        assert "eval-base" in content
        assert "eval-comp" in content

    def test_html_contains_dark_theme_vars(self, tmp_path):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(),
        )
        from raki.report.html_report import write_diff_html_report

        html_path = tmp_path / "diff.html"
        write_diff_html_report(diff, html_path)
        content = html_path.read_text()
        assert "--bg-primary" in content
        assert "--bg-secondary" in content

    def test_html_contains_transitions(self, tmp_path):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(
                matched_ids={"s1"},
                baseline_total=1,
                compare_total=1,
            ),
            regressions=[
                SessionTransition(
                    session_id="s1",
                    old_verdict="pass",
                    new_verdict="fail",
                    transition_type="regression",
                ),
            ],
        )
        from raki.report.html_report import write_diff_html_report

        html_path = tmp_path / "diff.html"
        write_diff_html_report(diff, html_path)
        content = html_path.read_text()
        assert "s1" in content
        assert "PASS" in content
        assert "FAIL" in content

    def test_html_contains_new_and_dropped(self, tmp_path):
        diff = DiffReport(
            baseline_run_id="base",
            compare_run_id="comp",
            match_result=MatchResult(
                matched_ids={"s1"},
                new_ids={"s2"},
                dropped_ids={"s3"},
                baseline_total=2,
                compare_total=2,
            ),
        )
        from raki.report.html_report import write_diff_html_report

        html_path = tmp_path / "diff.html"
        write_diff_html_report(diff, html_path)
        content = html_path.read_text()
        assert "s2" in content  # new session
        assert "s3" in content  # dropped session
        assert "dropped" in content.lower()


class TestMetricDeltaDirection:
    def test_cost_efficiency_decrease_is_improved(self):
        """Cost efficiency: lower is better (it's not in higher_is_better)."""
        baseline_scores = {"cost_efficiency": 12.30}
        compare_scores = {"cost_efficiency": 7.42}
        deltas = compute_deltas(baseline_scores, compare_scores)
        # cost_efficiency has higher_is_better=False in metadata
        assert deltas[0].direction == "improved"

    def test_unknown_metric_defaults_to_higher_is_better(self):
        baseline_scores = {"some_new_metric": 0.5}
        compare_scores = {"some_new_metric": 0.7}
        deltas = compute_deltas(baseline_scores, compare_scores)
        assert deltas[0].direction == "improved"
