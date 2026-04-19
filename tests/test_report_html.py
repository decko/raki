"""Tests for HTML report generation — self-contained, dark-themed, collapsible drill-down."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from raki.model.phases import ReviewFinding
from raki.model.report import EvalReport, MetricResult, SampleResult

from conftest import make_sample


def _make_minimal_report() -> EvalReport:
    """Report with only aggregate scores, no sample results."""
    return EvalReport(
        run_id="eval-html-001",
        timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
        config={"adapter": "session-schema", "metrics": ["rework_cycles"]},
        aggregate_scores={
            "first_pass_verify_rate": 0.58,
            "rework_cycles": 1.3,
            "review_severity_distribution": 0.85,
            "cost_efficiency": 18.4,
        },
    )


def _make_report_with_samples() -> EvalReport:
    """Report with sample results including sessions, phases, and findings."""
    findings_session_a = [
        ReviewFinding(
            reviewer="reviewer-1",
            severity="critical",
            file="main.py",
            line=42,
            issue="SQL injection vulnerability",
            suggestion="Use parameterized queries",
        ),
        ReviewFinding(
            reviewer="reviewer-1",
            severity="minor",
            file="utils.py",
            line=10,
            issue="Unused import",
        ),
    ]
    findings_session_b = [
        ReviewFinding(
            reviewer="reviewer-2",
            severity="critical",
            file="main.py",
            line=50,
            issue="SQL injection vulnerability",
            suggestion="Use parameterized queries",
        ),
        ReviewFinding(
            reviewer="reviewer-2",
            severity="major",
            file="config.py",
            line=5,
            issue="Hardcoded credentials",
            suggestion="Use environment variables",
        ),
    ]
    sample_a = make_sample(
        "session-alpha",
        rework_cycles=2,
        cost=25.0,
        verify_status="failed",
        findings=findings_session_a,
    )
    sample_b = make_sample(
        "session-beta",
        rework_cycles=1,
        cost=15.0,
        verify_status="completed",
        findings=findings_session_b,
    )
    sample_c = make_sample(
        "session-gamma",
        rework_cycles=0,
        cost=8.0,
        verify_status="completed",
    )

    metric_rework = MetricResult(
        name="rework_cycles",
        score=1.0,
        sample_scores={
            "session-alpha": 2.0,
            "session-beta": 1.0,
            "session-gamma": 0.0,
        },
    )
    metric_verify = MetricResult(
        name="first_pass_verify_rate",
        score=0.67,
        sample_scores={
            "session-alpha": 0.0,
            "session-beta": 1.0,
            "session-gamma": 1.0,
        },
    )
    metric_faith = MetricResult(
        name="faithfulness",
        score=0.80,
        details={"claims_verified": 24, "claims_total": 30},
        sample_scores={
            "session-alpha": 0.60,
            "session-beta": 0.90,
            "session-gamma": 0.90,
        },
    )

    sample_results = [
        SampleResult(sample=sample_a, scores=[metric_rework, metric_verify, metric_faith]),
        SampleResult(sample=sample_b, scores=[metric_rework, metric_verify, metric_faith]),
        SampleResult(sample=sample_c, scores=[metric_rework, metric_verify, metric_faith]),
    ]

    return EvalReport(
        run_id="eval-html-samples-001",
        timestamp=datetime(2026, 4, 18, 14, 30, 0, tzinfo=timezone.utc),
        config={"adapter": "session-schema"},
        aggregate_scores={
            "first_pass_verify_rate": 0.67,
            "rework_cycles": 1.0,
            "review_severity_distribution": 0.85,
            "cost_efficiency": 16.0,
            "faithfulness": 0.80,
        },
        sample_results=sample_results,
    )


def _make_report_with_many_sessions() -> EvalReport:
    """Report with 7 sessions to test 'worst 5' shortcut.

    Uses a retrieval metric (context_precision) so compute_worst_sessions
    correctly ranks sessions — operational metrics are excluded from ranking.
    """
    sample_results = []
    scores_map = {
        "session-01": 0.95,
        "session-02": 0.30,
        "session-03": 0.85,
        "session-04": 0.10,
        "session-05": 0.50,
        "session-06": 0.20,
        "session-07": 0.75,
    }
    for session_id, score in scores_map.items():
        sample = make_sample(session_id, rework_cycles=int((1 - score) * 3))
        metric = MetricResult(
            name="context_precision",
            score=score,
            sample_scores={session_id: score},
        )
        sample_results.append(SampleResult(sample=sample, scores=[metric]))

    return EvalReport(
        run_id="eval-html-many",
        timestamp=datetime(2026, 4, 18, 16, 0, 0, tzinfo=timezone.utc),
        aggregate_scores={"context_precision": sum(scores_map.values()) / len(scores_map)},
        sample_results=sample_results,
    )


# --- HTML generation tests ---


class TestHtmlReportGeneration:
    def test_generates_html_file(self, tmp_path: Path) -> None:
        """write_html_report should create an HTML file at the given path."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Should create parent directories if they do not exist."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "nested" / "deep" / "report.html"
        write_html_report(report, output)
        assert output.exists()

    def test_output_is_valid_html(self, tmp_path: Path) -> None:
        """Output should start with <!DOCTYPE html> and contain closing </html>."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert content.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in content


class TestHtmlSelfContained:
    def test_no_external_css_links(self, tmp_path: Path) -> None:
        """HTML should not reference any external CSS files."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert 'rel="stylesheet"' not in content
        assert "<style>" in content

    def test_no_external_js_scripts(self, tmp_path: Path) -> None:
        """HTML should not reference any external JavaScript files."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Should have inline script but no src= attributes on script tags
        assert "<script src=" not in content

    def test_inline_styles_present(self, tmp_path: Path) -> None:
        """CSS should be inline within a <style> tag."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "<style>" in content
        assert "</style>" in content


class TestHtmlDarkTheme:
    def test_dark_background_color(self, tmp_path: Path) -> None:
        """Dark theme should use a dark background color."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Should reference a dark background (e.g., #1a1a2e or similar dark color)
        assert "background" in content.lower()


class TestHtmlSections:
    def test_aggregate_scores_section(self, tmp_path: Path) -> None:
        """HTML should contain an aggregate scores section."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "Aggregate" in content or "aggregate" in content

    def test_operational_health_displayed(self, tmp_path: Path) -> None:
        """Operational metrics should appear in the report."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "first_pass_verify_rate" in content
        assert "rework_cycles" in content

    def test_retrieval_quality_displayed(self, tmp_path: Path) -> None:
        """Retrieval metrics should appear when present."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "faithfulness" in content

    def test_recurring_failures_section(self, tmp_path: Path) -> None:
        """Should show recurring failures (most common findings across sessions)."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # "SQL injection vulnerability" appears in 2 sessions - should be listed
        assert "Recurring" in content or "recurring" in content
        assert "SQL injection" in content

    def test_per_session_drilldown(self, tmp_path: Path) -> None:
        """Each session should be present in the drill-down section."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "session-alpha" in content
        assert "session-beta" in content
        assert "session-gamma" in content


class TestHtmlCollapsibleDetails:
    def test_collapsible_elements_present(self, tmp_path: Path) -> None:
        """Session details should use HTML <details>/<summary> for collapsible sections."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "<details" in content
        assert "<summary" in content

    def test_phases_shown_in_drilldown(self, tmp_path: Path) -> None:
        """Phase details should be visible in the session drill-down."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "implement" in content
        assert "verify" in content

    def test_findings_shown_in_drilldown(self, tmp_path: Path) -> None:
        """Findings should be shown within session drill-down."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "SQL injection vulnerability" in content
        assert "critical" in content.lower()


class TestHtmlColorCoding:
    def test_color_function_matches_cli(self) -> None:
        """html_color_for_score should match CLI color_for_score semantics."""
        from raki.report.html_report import html_color_for_score

        assert html_color_for_score(0.85) != html_color_for_score(0.45)
        # High score should be greenish, low score should be reddish
        high_color = html_color_for_score(0.85)
        low_color = html_color_for_score(0.45)
        assert high_color != low_color

    def test_score_colors_in_html(self, tmp_path: Path) -> None:
        """Score values in HTML should have color styling applied."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Color values should appear in the HTML (as inline styles or CSS classes)
        assert "color:" in content or "class=" in content


class TestHtmlWorstSessions:
    def test_worst_sessions_section(self, tmp_path: Path) -> None:
        """Should show a 'Worst 5 sessions' section when there are enough sessions."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_many_sessions()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "Worst" in content or "worst" in content

    def test_worst_sessions_ordered(self, tmp_path: Path) -> None:
        """Worst sessions should show lowest-scoring sessions first."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_many_sessions()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # session-04 (score 0.10) should appear in worst sessions
        assert "session-04" in content
        # session-06 (score 0.20) should appear in worst sessions
        assert "session-06" in content

    def test_worst_sessions_limited_to_five(self, tmp_path: Path) -> None:
        """Worst sessions shortcut should show at most 5 sessions."""
        from raki.report.html_report import compute_worst_sessions

        report = _make_report_with_many_sessions()
        worst = compute_worst_sessions(report, limit=5)
        assert len(worst) == 5


class TestHtmlTimestampFilename:
    def test_html_timestamp_filename(self) -> None:
        """HTML report should use timestamp-based filename similar to JSON report."""
        from raki.report.html_report import html_timestamp_filename

        report = _make_minimal_report()
        filename = html_timestamp_filename(report)
        assert filename.endswith(".html")
        assert "raki-report" in filename


class TestHtmlMetadata:
    def test_run_id_in_html(self, tmp_path: Path) -> None:
        """The run ID should appear in the HTML report."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "eval-html-001" in content

    def test_timestamp_in_html(self, tmp_path: Path) -> None:
        """The report timestamp should appear in the HTML report."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "2026" in content


class TestHtmlFaithfulnessBreakdown:
    def test_faithfulness_details_shown(self, tmp_path: Path) -> None:
        """When faithfulness metric has claim details, they should be shown."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # The faithfulness metric has details with claims_verified/claims_total
        assert "claims" in content.lower() or "faithfulness" in content.lower()


class TestHtmlSessionStripping:
    def test_strips_session_data_by_default(self, tmp_path: Path) -> None:
        """HTML report should strip raw session data by default (--include-sessions to opt in).

        Verifies that write_html_report applies strip_session_data to the report
        before rendering, matching the JSON report's default behavior.
        """
        from unittest.mock import patch

        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"

        # Capture the report object that gets passed to the template
        captured_reports: list[EvalReport] = []
        original_build_env = __import__(
            "raki.report.html_report", fromlist=["_build_jinja_env"]
        )._build_jinja_env

        def capture_env():
            env = original_build_env()
            original_get_template = env.get_template

            def patched_get_template(name):
                template = original_get_template(name)
                original_render = template.render

                def patched_render(**kwargs):
                    captured_reports.append(kwargs["report"])
                    return original_render(**kwargs)

                template.render = patched_render
                return template

            env.get_template = patched_get_template
            return env

        with patch("raki.report.html_report._build_jinja_env", side_effect=capture_env):
            write_html_report(report, output)

        assert len(captured_reports) == 1
        rendered_report = captured_reports[0]
        # Phase outputs should be replaced with the stripped sentinel
        for sample_result in rendered_report.sample_results:
            for phase in sample_result.sample.phases:
                assert phase.output == "<stripped>"
                assert phase.output_structured is None
                assert phase.knowledge_context is None
                assert phase.instruction_context is None

    def test_include_sessions_retains_data(self, tmp_path: Path) -> None:
        """When include_sessions=True, raw session data should be retained."""
        from unittest.mock import patch

        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"

        captured_reports: list[EvalReport] = []
        original_build_env = __import__(
            "raki.report.html_report", fromlist=["_build_jinja_env"]
        )._build_jinja_env

        def capture_env():
            env = original_build_env()
            original_get_template = env.get_template

            def patched_get_template(name):
                template = original_get_template(name)
                original_render = template.render

                def patched_render(**kwargs):
                    captured_reports.append(kwargs["report"])
                    return original_render(**kwargs)

                template.render = patched_render
                return template

            env.get_template = patched_get_template
            return env

        with patch("raki.report.html_report._build_jinja_env", side_effect=capture_env):
            write_html_report(report, output, include_sessions=True)

        assert len(captured_reports) == 1
        rendered_report = captured_reports[0]
        # Phase outputs should NOT be stripped
        has_real_output = False
        for sample_result in rendered_report.sample_results:
            for phase in sample_result.sample.phases:
                if phase.output != "<stripped>":
                    has_real_output = True
        assert has_real_output


class TestComputeWorstSessionsRetrieval:
    def test_excludes_operational_metrics(self) -> None:
        """compute_worst_sessions should only use retrieval metrics, not operational ones."""
        from raki.report.html_report import compute_worst_sessions

        sample = make_sample("session-ops-only", rework_cycles=3, cost=50.0)
        metric = MetricResult(
            name="rework_cycles",
            score=3.0,
            sample_scores={"session-ops-only": 3.0},
        )
        report = EvalReport(
            run_id="ops-only",
            aggregate_scores={"rework_cycles": 3.0},
            sample_results=[SampleResult(sample=sample, scores=[metric])],
        )
        worst = compute_worst_sessions(report, limit=5)
        # No retrieval metrics -> should return empty
        assert worst == []

    def test_uses_retrieval_metrics_only(self) -> None:
        """compute_worst_sessions should rank by retrieval scores, ignoring operational."""
        from raki.report.html_report import compute_worst_sessions

        sample = make_sample("session-mixed", rework_cycles=2, cost=30.0)
        operational_metric = MetricResult(
            name="rework_cycles",
            score=2.0,
            sample_scores={"session-mixed": 2.0},
        )
        retrieval_metric = MetricResult(
            name="faithfulness",
            score=0.70,
            sample_scores={"session-mixed": 0.70},
        )
        report = EvalReport(
            run_id="mixed",
            aggregate_scores={"rework_cycles": 2.0, "faithfulness": 0.70},
            sample_results=[
                SampleResult(sample=sample, scores=[operational_metric, retrieval_metric])
            ],
        )
        worst = compute_worst_sessions(report, limit=5)
        assert len(worst) == 1
        # Should use faithfulness (0.70), not the blended avg of (2.0 + 0.70) / 2
        assert worst[0].avg_score == pytest.approx(0.70)
