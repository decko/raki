"""Tests for HTML report generation — self-contained, dark-themed, collapsible drill-down."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from raki.model.dataset import EvalSample, SessionMeta
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
        """Operational metrics should appear in the report using display names."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "Verify rate" in content
        assert "Rework cycles" in content

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


# --- Issue #33: HTML report fixes ---


class TestHtmlColorHigherIsBetter:
    """html_color_for_score respects higher_is_better for inverted metrics."""

    def test_inverted_metric_low_score_is_green(self) -> None:
        """For inverted metrics (higher_is_better=False), low scores should be green."""
        from raki.report.html_report import html_color_for_score

        assert html_color_for_score(0.1, higher_is_better=False) == "green"

    def test_inverted_metric_high_score_is_red(self) -> None:
        """For inverted metrics (higher_is_better=False), high scores should be red."""
        from raki.report.html_report import html_color_for_score

        assert html_color_for_score(0.8, higher_is_better=False) == "red"

    def test_inverted_metric_mid_score_is_yellow(self) -> None:
        """For inverted metrics (higher_is_better=False), mid scores should be yellow."""
        from raki.report.html_report import html_color_for_score

        assert html_color_for_score(0.3, higher_is_better=False) == "yellow"

    def test_non_normalized_currency_returns_white(self) -> None:
        """Non-normalized metrics (currency) should return white, not a color band."""
        from raki.report.html_report import html_color_for_score

        assert (
            html_color_for_score(18.4, higher_is_better=False, display_format="currency") == "white"
        )

    def test_non_normalized_count_returns_white(self) -> None:
        """Non-normalized metrics (count) should return white, not a color band."""
        from raki.report.html_report import html_color_for_score

        assert html_color_for_score(3.0, higher_is_better=False, display_format="count") == "white"

    def test_inverted_metric_colors_in_report(self, tmp_path: Path) -> None:
        """Inverted metrics like knowledge_miss_rate should show correct colors."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-inverted",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={
                "knowledge_miss_rate": 0.1,  # Low miss rate = good = green
            },
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Miss rate 0.1 with higher_is_better=False should be green
        assert "color-green" in content


class TestProgressBarOmission:
    """Progress bars omitted for non-normalized metrics (cost, rework count)."""

    def test_no_progress_bar_for_cost(self, tmp_path: Path) -> None:
        """Cost metric should not have a progress bar since it's not 0-1 normalized."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-cost",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"cost_efficiency": 18.4},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # The cost card should not have a metric-bar element
        # Look for the display name "Cost / session" which is now used in the template
        cost_idx = content.find("Cost / session")
        assert cost_idx != -1, "Expected 'Cost / session' display name in HTML"
        # Find the end of this score-card div
        card_end = content.find("</div>", cost_idx + 50)
        next_card_end = content.find("</div>", card_end + 1)
        cost_section = (
            content[cost_idx:next_card_end] if next_card_end != -1 else content[cost_idx:]
        )
        assert "metric-bar" not in cost_section

    def test_no_progress_bar_for_rework_count(self, tmp_path: Path) -> None:
        """Rework cycles metric should not have a progress bar since it's a count."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-rework",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"rework_cycles": 1.3},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # rework_cycles card should not have a metric-bar — uses display name "Rework cycles"
        rework_idx = content.find("Rework cycles")
        assert rework_idx != -1, "Expected 'Rework cycles' display name in HTML"
        card_end = content.find("</div>", rework_idx + 50)
        next_card_end = content.find("</div>", card_end + 1)
        rework_section = (
            content[rework_idx:next_card_end] if next_card_end != -1 else content[rework_idx:]
        )
        assert "metric-bar" not in rework_section

    def test_progress_bar_present_for_normalized_metric(self, tmp_path: Path) -> None:
        """Normalized metrics (like verify rate, 0-1) should still have progress bars."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-verify",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "metric-bar" in content


class TestSessionCount:
    """session_count passed as separate template variable, not derived from sample_results."""

    def test_session_count_in_header(self, tmp_path: Path) -> None:
        """Session count should appear in the header from the session_count variable."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-count",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
            config={"session_count": 42},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output, session_count=42)
        content = output.read_text()
        assert "42" in content

    def test_session_count_defaults_to_sample_results_length(self, tmp_path: Path) -> None:
        """When session_count not provided, it should fall back to sample_results length."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # 3 sample results in _make_report_with_samples
        assert ">3<" in content or ">3 " in content or "Sessions:</strong> 3" in content


class TestEmptyStates:
    """All conditional sections have .empty-state messages when data is empty."""

    def test_no_retrieval_metrics_shows_empty_state(self, tmp_path: Path) -> None:
        """When there are no retrieval metrics, show an empty-state message."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-no-retrieval",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={
                "first_pass_verify_rate": 0.85,
                "rework_cycles": 1.0,
            },
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "empty-state" in content
        assert "Retrieval metrics require LLM judge" in content

    def test_no_recurring_failures_shows_empty_state(self, tmp_path: Path) -> None:
        """When there are no recurring failures, show an empty-state message."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "empty-state" in content
        # Should have some message about no recurring failures
        assert "No recurring" in content or "no recurring" in content

    def test_no_drilldown_shows_empty_state(self, tmp_path: Path) -> None:
        """When sample_results is empty, the drill-down section should show empty-state."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "empty-state" in content

    def test_no_worst_sessions_shows_empty_state(self, tmp_path: Path) -> None:
        """When there are no worst sessions, show an empty-state message."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "empty-state" in content


class TestDisplayNames:
    """display_name used on score cards, raw metric names in JSON only."""

    def test_display_name_on_score_card(self, tmp_path: Path) -> None:
        """Score cards should show display_name (e.g., 'Verify rate') not raw name."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-display",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={
                "first_pass_verify_rate": 0.85,
                "cost_efficiency": 7.74,
            },
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Should show display names
        assert "Verify rate" in content
        assert "Cost / session" in content

    def test_metric_description_subtitle(self, tmp_path: Path) -> None:
        """Score cards should show plain-English subtitle from METRIC_METADATA."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-desc",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Subtitle text (Jinja2 autoescape converts ' to &#39;)
        assert "passes all checks on the first try" in content


class TestAccessibility:
    """Accessibility improvements: button, aria-expanded, main landmark."""

    def test_main_landmark_present(self, tmp_path: Path) -> None:
        """HTML should use a <main> landmark for the main content area."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "<main" in content
        assert "</main>" in content

    def test_button_for_worst_session_rows(self, tmp_path: Path) -> None:
        """Worst session rows should use <button> instead of <a> for clickable rows."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_many_sessions()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Should use button, not anchor
        assert "<button" in content
        assert 'class="worst-session-row"' in content or "worst-session-row" in content
        # Should NOT use <a> for worst session rows
        assert '<a class="worst-session-row"' not in content

    def test_aria_expanded_on_collapsibles(self, tmp_path: Path) -> None:
        """Collapsible sections should have aria-expanded attributes."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "aria-expanded" in content


class TestCostFormatting:
    """Cost should display as $7.74, not 7.74."""

    def test_cost_displays_with_dollar_sign(self, tmp_path: Path) -> None:
        """Cost values should be formatted with a dollar sign prefix."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-cost-fmt",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"cost_efficiency": 7.74},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "$7.74" in content

    def test_cost_in_drilldown_has_dollar_sign(self, tmp_path: Path) -> None:
        """Cost in the per-session drill-down should also show dollar sign."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # The drill-down already shows $25.00 etc. - verify this is still the case
        assert "$25.00" in content or "$15.00" in content


class TestMetricMetadataSync:
    """METRIC_METADATA stays in sync with actual metric class attributes."""

    def test_metadata_matches_metric_classes(self) -> None:
        """Every metric class's display_name, higher_is_better, and display_format
        must match the corresponding entry in METRIC_METADATA."""
        from raki.metrics.knowledge import ALL_KNOWLEDGE
        from raki.metrics.operational import ALL_OPERATIONAL
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric
        from raki.metrics.ragas.precision import ContextPrecisionMetric
        from raki.metrics.ragas.recall import ContextRecallMetric
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric
        from raki.report.html_report import METRIC_METADATA

        all_metrics = (
            list(ALL_OPERATIONAL)
            + list(ALL_KNOWLEDGE)
            + [
                FaithfulnessMetric(),
                AnswerRelevancyMetric(),
                ContextPrecisionMetric(),
                ContextRecallMetric(),
            ]
        )

        for metric in all_metrics:
            meta = METRIC_METADATA.get(metric.name)
            assert meta is not None, f"Metric '{metric.name}' missing from METRIC_METADATA"
            assert meta["display_name"] == metric.display_name, (
                f"display_name mismatch for '{metric.name}': "
                f"metadata={meta['display_name']!r}, class={metric.display_name!r}"
            )
            assert meta["higher_is_better"] == metric.higher_is_better, (
                f"higher_is_better mismatch for '{metric.name}': "
                f"metadata={meta['higher_is_better']!r}, class={metric.higher_is_better!r}"
            )
            assert meta["display_format"] == metric.display_format, (
                f"display_format mismatch for '{metric.name}': "
                f"metadata={meta['display_format']!r}, class={metric.display_format!r}"
            )

    def test_all_metadata_entries_have_metric_class(self) -> None:
        """Every key in METRIC_METADATA must correspond to an actual metric class."""
        from raki.metrics.knowledge import ALL_KNOWLEDGE
        from raki.metrics.operational import ALL_OPERATIONAL
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric
        from raki.metrics.ragas.precision import ContextPrecisionMetric
        from raki.metrics.ragas.recall import ContextRecallMetric
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric
        from raki.report.html_report import METRIC_METADATA

        all_metrics = (
            list(ALL_OPERATIONAL)
            + list(ALL_KNOWLEDGE)
            + [
                FaithfulnessMetric(),
                AnswerRelevancyMetric(),
                ContextPrecisionMetric(),
                ContextRecallMetric(),
            ]
        )
        metric_names = {metric.name for metric in all_metrics}

        for metadata_key in METRIC_METADATA:
            assert metadata_key in metric_names, (
                f"METRIC_METADATA key '{metadata_key}' has no corresponding metric class"
            )


class TestPercentFormat:
    """display_format='percent' renders as '85%' not '0.85'."""

    def test_verify_rate_shows_percentage(self, tmp_path: Path) -> None:
        """first_pass_verify_rate (display_format=percent) should render as percentage."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-percent",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "85%" in content
        # Should NOT show raw 0.85 in the metric value
        # (0.85 may appear elsewhere, e.g. in a bar width, so we check the metric-value div)
        verify_idx = content.find("Verify rate")
        assert verify_idx != -1
        value_start = content.find("metric-value", verify_idx)
        value_end = content.find("</div>", value_start)
        value_section = content[value_start:value_end]
        assert "85%" in value_section


# --- Issue #70: Score cards redesign tests ---


def _make_session_meta_helper(session_id: str) -> SessionMeta:
    """Helper to create a SessionMeta for tests."""
    return SessionMeta(
        session_id=session_id,
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )


class TestSummarySentenceInHtml:
    """Summary sentence above score cards."""

    def test_summary_sentence_present(self, tmp_path: Path) -> None:
        """HTML report should include a summary sentence above the score cards."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "summary-sentence" in content

    def test_summary_sentence_contains_verify_rate(self, tmp_path: Path) -> None:
        """Summary sentence in HTML should contain the verify rate percentage."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # 0.67 = 67%
        assert "67%" in content


class TestHeroCard:
    """Verify rate should render as a hero card (wider, larger)."""

    def test_hero_card_class_present(self, tmp_path: Path) -> None:
        """Verify rate card should have a hero-card CSS class."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-hero",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.91},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "hero-card" in content


class TestPlainEnglishSubtitles:
    """Every card has a plain-English subtitle."""

    def test_verify_rate_subtitle(self, tmp_path: Path) -> None:
        """Verify rate card should have the plain-English subtitle."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-subtitles",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "passes all checks on the first try" in content

    def test_rework_cycles_subtitle(self, tmp_path: Path) -> None:
        """Rework cycles card should have the plain-English subtitle."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-subtitles",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"rework_cycles": 1.0},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "redo its work after feedback" in content

    def test_cost_subtitle(self, tmp_path: Path) -> None:
        """Cost card should have the plain-English subtitle."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-subtitles",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"cost_efficiency": 10.0},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "costs in API fees" in content


class TestDirectionBadges:
    """Direction badges on directional metrics (higher/lower is better)."""

    def test_verify_rate_shows_higher_is_better(self, tmp_path: Path) -> None:
        """Verify rate should show 'higher is better' direction badge."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-direction",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "higher is better" in content.lower()

    def test_rework_cycles_shows_lower_is_better(self, tmp_path: Path) -> None:
        """Rework cycles should show 'lower is better' direction badge."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-direction",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"rework_cycles": 1.0},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "lower is better" in content.lower()

    def test_verify_rate_shows_target_threshold(self, tmp_path: Path) -> None:
        """Verify rate card should show target threshold of >85%."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-threshold",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "&gt;85%" in content or ">85%" in content


class TestSeverityDistributionBar:
    """Severity as stacked distribution bar with counts + traffic-light label."""

    def test_severity_bar_present(self, tmp_path: Path) -> None:
        """Severity section should show a distribution bar, not a single score."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "severity-distribution" in content or "severity-bar" in content

    def test_severity_counts_shown(self, tmp_path: Path) -> None:
        """Severity distribution should show counts for each severity level."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # _make_report_with_samples has 2 critical, 1 major, 1 minor findings
        assert "critical" in content.lower()
        assert "major" in content.lower()
        assert "minor" in content.lower()

    def test_severity_label_shown(self, tmp_path: Path) -> None:
        """Severity distribution should show a traffic-light label."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        content_lower = content.lower()
        assert "clean" in content_lower or "moderate" in content_lower or "severe" in content_lower


class TestSeverityDistributionCompute:
    """compute_severity_distribution helper tests."""

    def test_clean_label_when_no_findings(self) -> None:
        """No findings at all should produce 'Clean' label."""
        from raki.report.html_report import SeverityDistribution, compute_severity_distribution

        report = EvalReport(
            run_id="clean",
            aggregate_scores={"review_severity_distribution": 1.0},
            sample_results=[
                SampleResult(sample=make_sample("s1"), scores=[]),
            ],
        )
        dist = compute_severity_distribution(report)
        assert isinstance(dist, SeverityDistribution)
        assert dist.label == "Clean"
        assert dist.critical == 0
        assert dist.major == 0
        assert dist.minor == 0

    def test_minor_label_with_only_major(self) -> None:
        """0 critical + some major = 'Minor' label."""
        from raki.report.html_report import compute_severity_distribution

        findings = [
            ReviewFinding(reviewer="r1", severity="major", issue="test issue"),
        ]
        report = EvalReport(
            run_id="minor-label",
            aggregate_scores={},
            sample_results=[
                SampleResult(sample=make_sample("s1", findings=findings), scores=[]),
            ],
        )
        dist = compute_severity_distribution(report)
        assert dist.label == "Minor"
        assert dist.critical == 0
        assert dist.major == 1

    def test_severe_label_when_weighted_above_threshold(self) -> None:
        """When weighted score > 0.5, label should be 'Severe'."""
        from raki.report.html_report import compute_severity_distribution

        findings = [
            ReviewFinding(reviewer="r1", severity="critical", issue="crit1"),
            ReviewFinding(reviewer="r1", severity="critical", issue="crit2"),
            ReviewFinding(reviewer="r1", severity="critical", issue="crit3"),
        ]
        report = EvalReport(
            run_id="severe-label",
            aggregate_scores={},
            sample_results=[
                SampleResult(sample=make_sample("s1", findings=findings), scores=[]),
            ],
        )
        dist = compute_severity_distribution(report)
        assert dist.label == "Severe"

    def test_moderate_label(self) -> None:
        """When there are some critical but weighted <= 0.5, label should be 'Moderate'."""
        from raki.report.html_report import compute_severity_distribution

        findings = [
            ReviewFinding(reviewer="r1", severity="critical", issue="crit1"),
            ReviewFinding(reviewer="r1", severity="minor", issue="minor1"),
            ReviewFinding(reviewer="r1", severity="minor", issue="minor2"),
            ReviewFinding(reviewer="r1", severity="minor", issue="minor3"),
            ReviewFinding(reviewer="r1", severity="minor", issue="minor4"),
            ReviewFinding(reviewer="r1", severity="minor", issue="minor5"),
        ]
        report = EvalReport(
            run_id="moderate-label",
            aggregate_scores={},
            sample_results=[
                SampleResult(sample=make_sample("s1", findings=findings), scores=[]),
            ],
        )
        dist = compute_severity_distribution(report)
        assert dist.label == "Moderate"


class TestKnowledgeMissRateConditional:
    """Knowledge Miss Rate hidden when no knowledge_context."""

    def test_hidden_when_no_knowledge_context(self, tmp_path: Path) -> None:
        """Knowledge Miss Rate renders as N/A when no session has knowledge_context.

        With the new behavior (issue #113), knowledge_miss_rate returns
        score=None when no knowledge_context exists, and the HTML template
        renders None scores as "N/A".
        """
        from raki.report.html_report import write_html_report

        sample = make_sample("s1")
        report = EvalReport(
            run_id="eval-no-knowledge",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={
                "first_pass_verify_rate": 0.85,
                "knowledge_miss_rate": None,
            },
            sample_results=[SampleResult(sample=sample, scores=[])],
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # With score=None, the template should render "N/A" for the metric
        assert "N/A" in content

    def test_shown_when_knowledge_context_present(self, tmp_path: Path) -> None:
        """Knowledge Miss Rate should be shown when sessions have knowledge_context."""
        from raki.model.phases import PhaseResult
        from raki.report.html_report import write_html_report

        meta = _make_session_meta_helper("s1")
        phases = [
            PhaseResult(
                name="implement",
                generation=1,
                status="completed",
                output="done",
                knowledge_context="some reference docs",
            ),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=[], events=[])
        report = EvalReport(
            run_id="eval-with-knowledge",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={
                "first_pass_verify_rate": 0.85,
                "knowledge_miss_rate": 0.15,
            },
            sample_results=[SampleResult(sample=sample, scores=[])],
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "Knowledge miss rate" in content or "Knowledge Miss Rate" in content


class TestRetrievalQualityConditional:
    """Retrieval Quality hidden in --no-llm mode."""

    def test_no_llm_footnote_when_no_retrieval(self, tmp_path: Path) -> None:
        """When has_retrieval=False, a footnote should appear instead of retrieval section."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-no-llm",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={
                "first_pass_verify_rate": 0.85,
            },
        )
        output = tmp_path / "report.html"
        write_html_report(report, output, has_retrieval=False)
        content = output.read_text()
        assert "Retrieval metrics omitted" in content or "without --no-llm" in content


class TestReworkCyclesColorThresholds:
    """Rework Cycles colored by threshold (green <1.0, yellow 1.0-2.0, red >2.0)."""

    def test_green_below_one(self, tmp_path: Path) -> None:
        """Rework cycles below 1.0 should be green."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-rework-green",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"rework_cycles": 0.5},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        rework_idx = content.find("Rework")
        assert rework_idx != -1
        card_section = content[rework_idx : rework_idx + 500]
        assert "color-green" in card_section or "rework-green" in card_section

    def test_yellow_between_one_and_two(self, tmp_path: Path) -> None:
        """Rework cycles between 1.0-2.0 should be yellow."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-rework-yellow",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"rework_cycles": 1.5},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        rework_idx = content.find("Rework")
        assert rework_idx != -1
        card_section = content[rework_idx : rework_idx + 500]
        assert "color-yellow" in card_section or "rework-yellow" in card_section

    def test_red_above_two(self, tmp_path: Path) -> None:
        """Rework cycles above 2.0 should be red."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-rework-red",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"rework_cycles": 2.5},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        rework_idx = content.find("Rework")
        assert rework_idx != -1
        card_section = content[rework_idx : rework_idx + 500]
        assert "color-red" in card_section or "rework-red" in card_section


class TestCostRange:
    """Cost shows min-max range."""

    def test_cost_range_shown(self, tmp_path: Path) -> None:
        """Cost card should show min-max range when samples exist."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "$8.00" in content or "8.00" in content
        assert "$25.00" in content or "25.00" in content


class TestComputeCostRange:
    """compute_cost_range helper tests."""

    def test_returns_min_max(self) -> None:
        """compute_cost_range should return min and max from sample costs."""
        from raki.report.html_report import compute_cost_range

        report = _make_report_with_samples()
        cost_min, cost_max = compute_cost_range(report)
        assert cost_min == pytest.approx(8.0)
        assert cost_max == pytest.approx(25.0)

    def test_returns_none_when_no_costs(self) -> None:
        """compute_cost_range should return None when no costs available."""
        from raki.report.html_report import compute_cost_range

        report = _make_minimal_report()
        result = compute_cost_range(report)
        assert result is None


class TestHasKnowledgeContext:
    """has_knowledge_context helper tests."""

    def test_false_when_no_context(self) -> None:
        """has_knowledge_context should return False when no phase has knowledge_context."""
        from raki.report.html_report import has_knowledge_context

        report = _make_report_with_samples()
        assert has_knowledge_context(report) is False

    def test_true_when_context_present(self) -> None:
        """has_knowledge_context should return True when any phase has knowledge_context."""
        from raki.model.phases import PhaseResult
        from raki.report.html_report import has_knowledge_context

        meta = _make_session_meta_helper("s1")
        phases = [
            PhaseResult(
                name="implement",
                generation=1,
                status="completed",
                output="done",
                knowledge_context="reference docs here",
            ),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=[], events=[])
        report = EvalReport(
            run_id="with-context",
            sample_results=[SampleResult(sample=sample, scores=[])],
        )
        assert has_knowledge_context(report) is True


class TestMetricLinksToInterpretingDocs:
    """Metric names link to docs/interpreting-results.md."""

    def test_metric_name_is_link(self, tmp_path: Path) -> None:
        """Metric names on score cards should be rendered as links."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="eval-links",
            timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            aggregate_scores={"first_pass_verify_rate": 0.85},
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "interpreting-results" in content
        assert "<a " in content


# --- Issue #68: Drill-down redesign tests ---


class TestDrillDownRowDataclass:
    """DrillDownRow frozen dataclass with verdict, detail, severity counts, cost, duration."""

    def test_drill_down_row_creation(self) -> None:
        """DrillDownRow should store session_id, verdict, detail, severity counts, cost, duration."""
        from raki.report.html_report import DrillDownRow

        row = DrillDownRow(
            session_id="session-101",
            verdict="fail",
            detail="implement failed",
            critical_count=2,
            major_count=0,
            minor_count=1,
            cost=4.20,
            duration_seconds=42,
            sort_key=(0, -4.20),
        )
        assert row.session_id == "session-101"
        assert row.verdict == "fail"
        assert row.detail == "implement failed"
        assert row.critical_count == 2
        assert row.major_count == 0
        assert row.minor_count == 1
        assert row.cost == pytest.approx(4.20)
        assert row.duration_seconds == 42
        assert row.sort_key == (0, -4.20)

    def test_drill_down_row_is_frozen(self) -> None:
        """DrillDownRow should be immutable (frozen dataclass)."""
        from raki.report.html_report import DrillDownRow

        row = DrillDownRow(
            session_id="session-101",
            verdict="pass",
            detail="5 phases",
            critical_count=0,
            major_count=0,
            minor_count=0,
            cost=8.50,
            duration_seconds=252,
            sort_key=(2, -8.50),
        )
        with pytest.raises(AttributeError):
            row.verdict = "fail"  # type: ignore[misc]


class TestDetermineVerdict:
    """determine_verdict: failed phase -> fail, rework_cycles > 0 -> rework, else pass."""

    def test_fail_when_phase_failed(self) -> None:
        """Session with a failed phase should have verdict 'fail'."""
        from raki.report.html_report import determine_verdict

        sample = make_sample("s1", verify_status="failed")
        assert determine_verdict(sample) == "fail"

    def test_rework_when_cycles_positive(self) -> None:
        """Session with rework_cycles > 0 but no failed phase should have verdict 'rework'."""
        from raki.report.html_report import determine_verdict

        sample = make_sample("s1", rework_cycles=2, verify_status="completed")
        assert determine_verdict(sample) == "rework"

    def test_pass_when_clean(self) -> None:
        """Session with no failed phases and 0 rework_cycles should have verdict 'pass'."""
        from raki.report.html_report import determine_verdict

        sample = make_sample("s1", rework_cycles=0, verify_status="completed")
        assert determine_verdict(sample) == "pass"

    def test_fail_takes_precedence_over_rework(self) -> None:
        """When both failed phase and rework_cycles > 0, verdict should be 'fail'."""
        from raki.report.html_report import determine_verdict

        sample = make_sample("s1", rework_cycles=2, verify_status="failed")
        assert determine_verdict(sample) == "fail"


class TestBuildDetail:
    """build_detail: "implement failed" / "2 cycles" / "5 phases"."""

    def test_detail_for_fail(self) -> None:
        """When a phase fails, detail should name the failed phase."""
        from raki.report.html_report import build_detail

        sample = make_sample("s1", verify_status="failed")
        detail = build_detail(sample)
        assert "verify failed" in detail

    def test_detail_for_rework(self) -> None:
        """When rework_cycles > 0 and no failure, detail should show cycle count."""
        from raki.report.html_report import build_detail

        sample = make_sample("s1", rework_cycles=2, verify_status="completed")
        detail = build_detail(sample)
        assert "2 cycles" in detail

    def test_detail_for_pass(self) -> None:
        """When clean pass, detail should show phase count."""
        from raki.report.html_report import build_detail

        sample = make_sample("s1", rework_cycles=0, verify_status="completed")
        detail = build_detail(sample)
        assert "phases" in detail


class TestComputeDuration:
    """_compute_duration: sum phase durations in seconds."""

    def test_sums_phase_durations(self) -> None:
        """Should sum all phase duration_ms values and convert to seconds."""
        from raki.model.phases import PhaseResult
        from raki.report.html_report import _compute_duration

        meta = SessionMeta(
            session_id="s1",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=2,
            rework_cycles=0,
        )
        phases = [
            PhaseResult(
                name="implement",
                generation=1,
                status="completed",
                output="done",
                duration_ms=60000,
            ),
            PhaseResult(
                name="verify",
                generation=1,
                status="completed",
                output="PASS",
                duration_ms=30000,
            ),
        ]
        sample = EvalSample(session=meta, phases=phases, findings=[], events=[])
        assert _compute_duration(sample) == 90

    def test_zero_when_no_duration_data(self) -> None:
        """Should return 0 when phases have no duration_ms."""
        from raki.report.html_report import _compute_duration

        sample = make_sample("s1")
        assert _compute_duration(sample) == 0


class TestFormatDuration:
    """_format_duration: format seconds as M:SS."""

    def test_format_minutes_seconds(self) -> None:
        """Should format 252 seconds as '4:12'."""
        from raki.report.html_report import _format_duration

        assert _format_duration(252) == "4:12"

    def test_format_zero(self) -> None:
        """Should format 0 seconds as '0:00'."""
        from raki.report.html_report import _format_duration

        assert _format_duration(0) == "0:00"

    def test_format_under_minute(self) -> None:
        """Should format 42 seconds as '0:42'."""
        from raki.report.html_report import _format_duration

        assert _format_duration(42) == "0:42"

    def test_format_exact_minutes(self) -> None:
        """Should format 120 seconds as '2:00'."""
        from raki.report.html_report import _format_duration

        assert _format_duration(120) == "2:00"

    def test_format_large_duration(self) -> None:
        """Should format 454 seconds as '7:34'."""
        from raki.report.html_report import _format_duration

        assert _format_duration(454) == "7:34"


class TestComputeDrillDownRows:
    """compute_drill_down_rows: build and sort rows with FAIL first, then REWORK, then PASS."""

    def test_sorts_fail_first(self) -> None:
        """FAIL rows should come before REWORK and PASS."""
        from raki.report.html_report import compute_drill_down_rows

        sample_fail = make_sample("s-fail", verify_status="failed", cost=4.20)
        sample_rework = make_sample("s-rework", rework_cycles=2, cost=26.10)
        sample_pass = make_sample("s-pass", rework_cycles=0, cost=8.50)
        sample_results = [
            SampleResult(sample=sample_pass, scores=[]),
            SampleResult(sample=sample_fail, scores=[]),
            SampleResult(sample=sample_rework, scores=[]),
        ]
        rows = compute_drill_down_rows(sample_results)
        assert rows[0].verdict == "fail"
        assert rows[1].verdict == "rework"
        assert rows[2].verdict == "pass"

    def test_cost_descending_within_verdict_group(self) -> None:
        """Within the same verdict group, rows should be sorted by cost descending."""
        from raki.report.html_report import compute_drill_down_rows

        sample_pass_cheap = make_sample("s-cheap", rework_cycles=0, cost=5.00)
        sample_pass_expensive = make_sample("s-expensive", rework_cycles=0, cost=20.00)
        sample_results = [
            SampleResult(sample=sample_pass_cheap, scores=[]),
            SampleResult(sample=sample_pass_expensive, scores=[]),
        ]
        rows = compute_drill_down_rows(sample_results)
        assert rows[0].session_id == "s-expensive"
        assert rows[1].session_id == "s-cheap"

    def test_severity_counts_from_findings(self) -> None:
        """Row severity counts should reflect actual findings."""
        from raki.report.html_report import compute_drill_down_rows

        findings = [
            ReviewFinding(reviewer="r1", severity="critical", issue="crit1"),
            ReviewFinding(reviewer="r1", severity="critical", issue="crit2"),
            ReviewFinding(reviewer="r1", severity="major", issue="maj1"),
        ]
        sample = make_sample("s1", rework_cycles=1, findings=findings)
        rows = compute_drill_down_rows([SampleResult(sample=sample, scores=[])])
        assert rows[0].critical_count == 2
        assert rows[0].major_count == 1
        assert rows[0].minor_count == 0

    def test_empty_sample_results_returns_empty(self) -> None:
        """Empty sample_results should return empty list."""
        from raki.report.html_report import compute_drill_down_rows

        rows = compute_drill_down_rows([])
        assert rows == []


class TestDrillDownRowsInHtml:
    """Drill-down rows render in HTML with verdict badges, borders, cost, and duration."""

    def test_verdict_badges_in_html(self, tmp_path: Path) -> None:
        """HTML drill-down should show FAIL/REWORK/PASS verdict badges."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # session-alpha has verify failed -> FAIL
        assert "verdict-fail" in content or "FAIL" in content
        # session-beta has rework_cycles=1 -> REWORK
        assert "verdict-rework" in content or "REWORK" in content
        # session-gamma has no rework/failure -> PASS
        assert "verdict-pass" in content or "PASS" in content

    def test_verdict_left_border_colors(self, tmp_path: Path) -> None:
        """Rows should have left border colored by verdict (red/yellow/none)."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "border-fail" in content or "border-left" in content

    def test_cost_shown_in_drilldown_row(self, tmp_path: Path) -> None:
        """Each drill-down row should show cost with dollar sign."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "$25.00" in content
        assert "$15.00" in content
        assert "$8.00" in content

    def test_severity_badges_on_fail_rework_only(self, tmp_path: Path) -> None:
        """Severity badges should appear on FAIL/REWORK rows but not on clean PASS rows."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # session-gamma is PASS with no findings -- should not have severity pills in row
        # Find the gamma session in the drill-down section (verdict-pass details element)
        drilldown_start = content.find("Per-Session Drill-Down")
        assert drilldown_start != -1
        gamma_idx = content.find("session-gamma", drilldown_start)
        assert gamma_idx != -1
        # Find the next summary/details boundary after gamma
        next_detail_idx = content.find("</summary>", gamma_idx)
        gamma_row = content[gamma_idx:next_detail_idx]
        assert "row-severity-pill-critical" not in gamma_row
        assert "row-severity-pill-major" not in gamma_row

    def test_drill_down_sorted_fail_first(self, tmp_path: Path) -> None:
        """In HTML, FAIL sessions should appear before REWORK which appear before PASS."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # session-alpha is FAIL (verify failed), session-beta is REWORK, session-gamma is PASS
        fail_pos = content.find("session-alpha")
        rework_pos = content.find("session-beta")
        pass_pos = content.find("session-gamma")
        assert fail_pos < rework_pos < pass_pos


class TestNeedsAttentionSection:
    """'Needs Attention' section above full drill-down with FAIL+REWORK count badge."""

    def test_needs_attention_present(self, tmp_path: Path) -> None:
        """When FAIL or REWORK sessions exist, 'Needs Attention' section should appear."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "needs-attention" in content.lower() or "Needs Attention" in content

    def test_needs_attention_count_badge(self, tmp_path: Path) -> None:
        """Needs Attention section should show count of FAIL+REWORK sessions."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # session-alpha (FAIL) + session-beta (REWORK) = 2
        assert "2 sessions need attention" in content.lower() or "2 session" in content.lower()

    def test_needs_attention_shows_only_fail_rework(self, tmp_path: Path) -> None:
        """Needs Attention section should only contain FAIL and REWORK sessions."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Find the Needs Attention heading in the HTML body (not CSS)
        attention_heading = content.find("Needs Attention")
        assert attention_heading != -1
        # The section should end before the full drill-down starts
        drilldown_idx = content.find("Per-Session Drill-Down", attention_heading)
        assert drilldown_idx != -1
        attention_section = content[attention_heading:drilldown_idx]
        # session-gamma (PASS) should NOT be in attention section
        assert "session-gamma" not in attention_section
        # session-alpha (FAIL) and session-beta (REWORK) should be present
        assert "session-alpha" in attention_section
        assert "session-beta" in attention_section

    def test_no_needs_attention_when_all_pass(self, tmp_path: Path) -> None:
        """When all sessions pass, Needs Attention section should not appear or be empty."""
        from raki.report.html_report import write_html_report

        all_pass_sample = make_sample("s1", rework_cycles=0, cost=10.0)
        report = EvalReport(
            run_id="all-pass",
            aggregate_scores={"first_pass_verify_rate": 1.0},
            sample_results=[SampleResult(sample=all_pass_sample, scores=[])],
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Either no needs-attention section, or it has an empty state
        attention_lower = content.lower()
        if "needs-attention" in attention_lower or "needs attention" in attention_lower:
            # It's acceptable if the section exists but has no rows
            assert "0 sessions" in attention_lower or "no sessions" in attention_lower


class TestDrillDownEmptyState:
    """Empty state when no session data."""

    def test_empty_state_message(self, tmp_path: Path) -> None:
        """When sample_results is empty, show message about --include-sessions."""
        from raki.report.html_report import write_html_report

        report = _make_minimal_report()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "--include-sessions" in content


class TestDrillDownExpandedView:
    """Expanded view: phase timeline with status dots + duration, findings with severity."""

    def test_phase_timeline_in_expanded_view(self, tmp_path: Path) -> None:
        """Expanded session should show phase timeline with status dots."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Phase status dots
        assert "phase-status" in content
        assert "implement" in content
        assert "verify" in content

    def test_findings_with_severity_in_expanded_view(self, tmp_path: Path) -> None:
        """Expanded session should show findings with severity badges and file location."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "SQL injection vulnerability" in content
        assert "main.py" in content
        assert "severity-critical" in content or "critical" in content.lower()

    def test_reviewer_name_in_expanded_view(self, tmp_path: Path) -> None:
        """Expanded session should show reviewer name for findings."""
        from raki.report.html_report import write_html_report

        report = _make_report_with_samples()
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        assert "reviewer-1" in content or "reviewer" in content.lower()


class TestPassRowsClean:
    """PASS rows should be visually clean -- no severity badges unless minor findings exist."""

    def test_pass_row_no_badges_when_clean(self, tmp_path: Path) -> None:
        """PASS row with no findings should have no severity badges in the summary."""
        from raki.report.html_report import write_html_report

        clean_pass = make_sample("s-clean", rework_cycles=0, cost=6.80)
        report = EvalReport(
            run_id="clean-pass",
            aggregate_scores={},
            sample_results=[SampleResult(sample=clean_pass, scores=[])],
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Find the summary row for s-clean
        clean_idx = content.find("s-clean")
        assert clean_idx != -1
        summary_end = content.find("</summary>", clean_idx)
        row_text = content[clean_idx:summary_end]
        # No severity pills in summary
        assert "critical" not in row_text.lower()
        assert "major" not in row_text.lower()

    def test_pass_row_shows_minor_badge_when_minor_findings(self, tmp_path: Path) -> None:
        """PASS row with minor findings should show minor badge."""
        from raki.report.html_report import write_html_report

        findings = [
            ReviewFinding(reviewer="r1", severity="minor", issue="Style issue"),
        ]
        minor_pass = make_sample("s-minor", rework_cycles=0, cost=8.50, findings=findings)
        report = EvalReport(
            run_id="minor-pass",
            aggregate_scores={},
            sample_results=[SampleResult(sample=minor_pass, scores=[])],
        )
        output = tmp_path / "report.html"
        write_html_report(report, output)
        content = output.read_text()
        # Find the summary row for s-minor
        minor_idx = content.find("s-minor")
        assert minor_idx != -1
        summary_end = content.find("</summary>", minor_idx)
        row_text = content[minor_idx:summary_end]
        assert "1 minor" in row_text or "minor" in row_text.lower()
