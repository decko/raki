"""Tests for report generation — CLI summary (Rich) and JSON serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest

from raki.model.phases import ReviewFinding
from raki.model.report import EvalReport, MetricResult, SampleResult
from raki.report.cli_summary import (
    color_for_score,
    format_metric_line,
    generate_summary_sentence,
    print_summary,
)
from raki.report.json_report import load_json_report, write_json_report

from conftest import make_dataset, make_sample


def _make_report() -> EvalReport:
    return EvalReport(
        run_id="eval-test-001",
        config={"adapter": "session-schema", "metrics": ["rework_cycles"]},
        aggregate_scores={
            "first_pass_verify_rate": 0.58,
            "rework_cycles": 1.3,
            "review_severity_distribution": 0.85,
            "cost_efficiency": 18.4,
        },
    )


def _make_report_with_samples() -> EvalReport:
    """Build a report that includes sample results with session data."""
    sample = make_sample("session-001", rework_cycles=1, cost=15.0)
    metric_result = MetricResult(
        name="rework_cycles",
        score=1.0,
        sample_scores={"session-001": 1.0},
    )
    sample_result = SampleResult(sample=sample, scores=[metric_result])
    return EvalReport(
        run_id="eval-with-sessions-001",
        config={"adapter": "session-schema"},
        aggregate_scores={"rework_cycles": 1.0},
        sample_results=[sample_result],
    )


# --- JSON round-trip tests ---


class TestJsonReportSizeLimit:
    def test_load_json_report_rejects_oversized_file(self, tmp_path: Path) -> None:
        """load_json_report() should reject files exceeding 50MB."""
        big_file = tmp_path / "big-report.json"
        big_file.write_text('{"run_id": "x"}' + " " * (50 * 1024 * 1024))
        with pytest.raises(ValueError, match="exceeding"):
            load_json_report(big_file)

    def test_load_json_report_accepts_normal_file(self, tmp_path: Path) -> None:
        """load_json_report() should accept files under the 50MB limit."""
        report = _make_report()
        output_path = tmp_path / "normal.json"
        write_json_report(report, output_path)
        loaded = load_json_report(output_path)
        assert loaded.run_id == report.run_id


class TestJsonReportSymlinkCheck:
    def test_load_json_report_rejects_symlink(self, tmp_path: Path) -> None:
        """load_json_report() should reject symlinked files."""
        report = _make_report()
        real_file = tmp_path / "real-report.json"
        write_json_report(report, real_file)
        link_file = tmp_path / "link-report.json"
        link_file.symlink_to(real_file)
        with pytest.raises(ValueError, match="symlink"):
            load_json_report(link_file)


class TestJsonReportRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        report = _make_report()
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        assert output_path.exists()
        loaded = load_json_report(output_path)
        assert loaded.run_id == "eval-test-001"
        assert loaded.aggregate_scores["rework_cycles"] == 1.3

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        report = _make_report()
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        raw = json.loads(output_path.read_text())
        assert "run_id" in raw
        assert "aggregate_scores" in raw

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        report = _make_report()
        output_path = tmp_path / "nested" / "deep" / "report.json"
        write_json_report(report, output_path)
        assert output_path.exists()

    def test_timestamp_preserved_after_round_trip(self, tmp_path: Path) -> None:
        report = _make_report()
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        loaded = load_json_report(output_path)
        assert loaded.timestamp.tzinfo is not None


class TestJsonReportStripping:
    def test_strips_session_data_by_default(self, tmp_path: Path) -> None:
        report = _make_report_with_samples()
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        raw = json.loads(output_path.read_text())
        sample = raw["sample_results"][0]["sample"]
        # Phase output should be replaced with sentinel (required field)
        for phase in sample["phases"]:
            assert phase["output"] == "<stripped>"
        # Optional sensitive fields should be removed entirely
        for phase in sample["phases"]:
            assert "output_structured" not in phase
            assert "knowledge_context" not in phase
            assert "instruction_context" not in phase
        # Events data should be stripped
        for event in sample.get("events", []):
            assert "data" not in event

    def test_stripped_report_round_trips_successfully(self, tmp_path: Path) -> None:
        """A stripped report must be loadable — required fields use sentinels, not removed."""
        report = _make_report_with_samples()
        output_path = tmp_path / "stripped.json"
        write_json_report(report, output_path)  # strips by default
        loaded = load_json_report(output_path)
        assert loaded.run_id == report.run_id
        for sample_result in loaded.sample_results:
            for phase in sample_result.sample.phases:
                assert phase.output == "<stripped>"

    def test_include_sessions_retains_data(self, tmp_path: Path) -> None:
        report = _make_report_with_samples()
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path, include_sessions=True)
        raw = json.loads(output_path.read_text())
        sample = raw["sample_results"][0]["sample"]
        # Phase output should be retained
        has_output = any("output" in phase for phase in sample["phases"])
        assert has_output


class TestJsonReportTimestampFilename:
    def test_timestamp_based_filename_avoids_overwrites(self, tmp_path: Path) -> None:
        """Timestamp-based filenames should use datetime, not date-only, to avoid overwrites."""
        report_a = EvalReport(
            run_id="eval-a",
            timestamp=datetime(2026, 4, 18, 10, 30, 0, tzinfo=timezone.utc),
            aggregate_scores={"metric": 0.5},
        )
        report_b = EvalReport(
            run_id="eval-b",
            timestamp=datetime(2026, 4, 18, 14, 45, 30, tzinfo=timezone.utc),
            aggregate_scores={"metric": 0.7},
        )
        from raki.report.json_report import timestamp_filename

        filename_a = timestamp_filename(report_a)
        filename_b = timestamp_filename(report_b)
        assert filename_a != filename_b
        # Filenames should contain time components, not just date
        assert "1030" in filename_a or "10-30" in filename_a or "T" in filename_a
        assert filename_a.endswith(".json")
        assert filename_b.endswith(".json")


# --- color_for_score tests ---


class TestColorForScore:
    def test_high_score_higher_is_better_green(self) -> None:
        assert color_for_score(0.85, higher_is_better=True) == "green"

    def test_medium_score_higher_is_better_yellow(self) -> None:
        assert color_for_score(0.65, higher_is_better=True) == "yellow"

    def test_low_score_higher_is_better_red(self) -> None:
        assert color_for_score(0.45, higher_is_better=True) == "red"

    def test_currency_lower_is_better_white(self) -> None:
        """Currency metrics where lower is better should be white, not colored."""
        assert color_for_score(18.4, higher_is_better=False, display_format="currency") == "white"

    def test_count_lower_is_better_white(self) -> None:
        """Count metrics where lower is better should be white, not colored."""
        assert color_for_score(1.3, higher_is_better=False, display_format="count") == "white"

    def test_inverted_scale_low_value_green(self) -> None:
        """For inverted score metrics (lower is better on 0-1 scale), low = green."""
        assert color_for_score(0.15, higher_is_better=False, display_format="score") == "green"

    def test_inverted_scale_high_value_red(self) -> None:
        assert color_for_score(0.75, higher_is_better=False, display_format="score") == "red"

    def test_inverted_scale_mid_value_yellow(self) -> None:
        assert color_for_score(0.35, higher_is_better=False, display_format="score") == "yellow"


# --- format_metric_line tests ---


class TestFormatMetricLine:
    def test_contains_name_and_score(self) -> None:
        line = format_metric_line("first_pass_verify_rate", 0.85, "(32/38 passed)")
        assert "first_pass_verify_rate" in line
        assert "0.85" in line

    def test_red_for_low_score(self) -> None:
        line = format_metric_line("first_pass_verify_rate", 0.45, "(17/38 passed)")
        assert "0.45" in line
        assert "[red]" in line

    def test_green_for_high_score(self) -> None:
        line = format_metric_line("first_pass_verify_rate", 0.85)
        assert "[green]" in line

    def test_currency_display_format(self) -> None:
        line = format_metric_line(
            "cost_efficiency",
            18.40,
            display_format="currency",
            higher_is_better=False,
        )
        assert "$18.40" in line

    def test_count_display_format(self) -> None:
        line = format_metric_line(
            "rework_cycles",
            1.3,
            display_format="count",
            higher_is_better=False,
        )
        assert "1.3" in line

    def test_sample_count_shown(self) -> None:
        line = format_metric_line("first_pass_verify_rate", 0.85, sample_count=38)
        assert "(n=38)" in line

    def test_display_name_used_when_provided(self) -> None:
        line = format_metric_line(
            "first_pass_verify_rate",
            0.85,
            display_name="First-Pass Verify Rate",
        )
        assert "First-Pass Verify Rate" in line
        assert "first_pass_verify_rate" not in line

    def test_percent_display_format(self) -> None:
        line = format_metric_line(
            "review_severity_distribution",
            0.85,
            display_format="percent",
        )
        assert "0.85" in line


# --- print_summary tests ---


class TestPrintSummary:
    def test_prints_operational_health_heading(self) -> None:
        """print_summary should show Operational Health as a category heading."""
        from io import StringIO

        from rich.console import Console

        report = _make_report()
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=38, console=test_console)
        output = string_io.getvalue()
        assert "Operational Health" in output

    def test_small_sample_caveat_below_50(self) -> None:
        """When n < 50, a caveat should appear alongside scores."""
        from io import StringIO

        from rich.console import Console

        report = _make_report()
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=30, console=test_console)
        output = string_io.getvalue()
        assert "Small sample size" in output or "small sample" in output.lower()

    def test_no_caveat_at_50_or_above(self) -> None:
        """When n >= 50, no small sample caveat should be shown."""
        from io import StringIO

        from rich.console import Console

        report = _make_report()
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=50, console=test_console)
        output = string_io.getvalue()
        assert "small sample" not in output.lower()

    def test_skipped_and_error_counts(self) -> None:
        """When there are skipped or errored sessions, counts should be shown."""
        from io import StringIO

        from rich.console import Console

        report = _make_report()
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(
            report, session_count=38, skipped_count=2, error_count=1, console=test_console
        )
        output = string_io.getvalue()
        assert "38" in output
        assert "skipped" in output.lower()

    def test_two_categories_displayed(self) -> None:
        """Both Operational Health and Retrieval Quality should appear when both exist."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="both-categories",
            aggregate_scores={
                "first_pass_verify_rate": 0.80,
                "faithfulness": 0.92,
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=100, console=test_console)
        output = string_io.getvalue()
        assert "Operational Health" in output
        assert "Retrieval Quality" in output

    def test_experimental_tag_for_faithfulness(self) -> None:
        """Faithfulness should show [experimental] tag in CLI summary."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="exp-faith",
            aggregate_scores={
                "faithfulness": 0.85,
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "experimental" in output.lower()

    def test_experimental_tag_for_answer_relevancy(self) -> None:
        """answer_relevancy should show [experimental] tag in CLI summary."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="exp-relevancy",
            aggregate_scores={
                "answer_relevancy": 0.72,
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "experimental" in output.lower()

    def test_calibration_caveat_for_retrieval_metrics(self) -> None:
        """When retrieval metrics are shown, a calibration caveat should appear."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="calibration-test",
            aggregate_scores={
                "context_precision": 0.80,
                "faithfulness": 0.85,
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "same-provider" in output.lower() or "llm judge" in output.lower()

    def test_no_calibration_caveat_when_no_retrieval(self) -> None:
        """When only operational metrics are shown, no calibration caveat should appear."""
        from io import StringIO

        from rich.console import Console

        report = _make_report()
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "same-provider" not in output.lower()


# --- generate_summary_sentence tests ---


def _make_finding(severity: Literal["critical", "major", "minor"]) -> ReviewFinding:
    """Helper to create a ReviewFinding with a given severity."""
    return ReviewFinding(
        reviewer="test-reviewer",
        severity=severity,
        issue=f"Test {severity} issue",
    )


class TestGenerateSummarySentence:
    def test_includes_verify_rate_percentage(self) -> None:
        """Summary sentence should include verify rate as a percentage."""
        report = EvalReport(
            run_id="summary-test",
            aggregate_scores={
                "first_pass_verify_rate": 0.91,
                "rework_cycles": 0.4,
                "cost_efficiency": 7.42,
            },
        )
        sentence = generate_summary_sentence(report, session_count=20)
        assert "91%" in sentence

    def test_includes_rework_cycles(self) -> None:
        """Summary sentence should include average rework cycles."""
        report = EvalReport(
            run_id="summary-test",
            aggregate_scores={
                "first_pass_verify_rate": 0.91,
                "rework_cycles": 0.4,
                "cost_efficiency": 7.42,
            },
        )
        sentence = generate_summary_sentence(report, session_count=20)
        assert "0.4" in sentence

    def test_includes_cost(self) -> None:
        """Summary sentence should include average cost per session."""
        report = EvalReport(
            run_id="summary-test",
            aggregate_scores={
                "first_pass_verify_rate": 0.91,
                "rework_cycles": 0.4,
                "cost_efficiency": 7.42,
            },
        )
        sentence = generate_summary_sentence(report, session_count=20)
        assert "$7.42" in sentence

    def test_includes_finding_counts_from_samples(self) -> None:
        """Summary sentence should include severity breakdown when samples exist."""
        findings_a = [
            _make_finding("critical"),
            _make_finding("critical"),
            _make_finding("major"),
        ]
        findings_b = [
            _make_finding("minor"),
            _make_finding("minor"),
            _make_finding("minor"),
        ]
        sample_a = make_sample("s1", findings=findings_a)
        sample_b = make_sample("s2", findings=findings_b)
        report = EvalReport(
            run_id="summary-findings",
            aggregate_scores={
                "first_pass_verify_rate": 0.80,
                "rework_cycles": 1.0,
                "cost_efficiency": 10.0,
            },
            sample_results=[
                SampleResult(sample=sample_a, scores=[]),
                SampleResult(sample=sample_b, scores=[]),
            ],
        )
        sentence = generate_summary_sentence(report, session_count=2)
        assert "2 critical" in sentence
        assert "1 major" in sentence
        assert "3 minor" in sentence

    def test_omits_missing_metrics(self) -> None:
        """Summary sentence should gracefully handle missing metrics."""
        report = EvalReport(
            run_id="summary-partial",
            aggregate_scores={
                "first_pass_verify_rate": 0.75,
            },
        )
        sentence = generate_summary_sentence(report, session_count=10)
        assert "75%" in sentence
        # Should not crash, should produce a reasonable sentence
        assert isinstance(sentence, str)
        assert len(sentence) > 0

    def test_returns_string(self) -> None:
        """generate_summary_sentence should always return a string."""
        report = EvalReport(
            run_id="summary-empty",
            aggregate_scores={},
        )
        sentence = generate_summary_sentence(report, session_count=0)
        assert isinstance(sentence, str)


# --- No-data metric display tests ---


class TestNoDataMetricDisplay:
    def test_format_metric_line_shows_na_when_no_data(self) -> None:
        line = format_metric_line(
            "token_efficiency",
            0.0,
            display_name="Tokens / phase",
            no_data=True,
        )
        assert "N/A" in line
        assert "no data" in line
        assert "[dim]" in line

    def test_format_metric_line_shows_score_when_data_exists(self) -> None:
        line = format_metric_line(
            "token_efficiency",
            1500.0,
            display_format="count",
            display_name="Tokens / phase",
            no_data=False,
        )
        assert "1500.0" in line
        assert "N/A" not in line

    def test_print_summary_shows_na_for_no_data_metric(self) -> None:
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="test-no-data",
            aggregate_scores={
                "first_pass_verify_rate": 0.8,
                "token_efficiency": 0.0,
            },
            metric_details={
                "first_pass_verify_rate": {"passed": 8, "total": 10},
                "token_efficiency": {
                    "mean_tokens_per_phase": 0.0,
                    "sessions_with_tokens": 0,
                },
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "N/A" in output

    def test_print_summary_shows_score_when_data_present(self) -> None:
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="test-with-data",
            aggregate_scores={
                "token_efficiency": 1500.0,
            },
            metric_details={
                "token_efficiency": {
                    "mean_tokens_per_phase": 1500.0,
                    "sessions_with_tokens": 5,
                },
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=5, console=test_console)
        output = string_io.getvalue()
        assert "N/A" not in output

    def test_has_no_data_detects_sessions_with_zero(self) -> None:
        from raki.report.cli_summary import _has_no_data

        details = {
            "token_efficiency": {"sessions_with_tokens": 0, "mean_tokens_per_phase": 0.0},
        }
        assert _has_no_data(details, "token_efficiency") is True

    def test_has_no_data_returns_false_when_sessions_present(self) -> None:
        from raki.report.cli_summary import _has_no_data

        details = {
            "token_efficiency": {"sessions_with_tokens": 5, "mean_tokens_per_phase": 1500.0},
        }
        assert _has_no_data(details, "token_efficiency") is False

    def test_has_no_data_returns_false_for_missing_metric(self) -> None:
        from raki.report.cli_summary import _has_no_data

        assert _has_no_data({}, "token_efficiency") is False

    def test_metric_details_populated_by_engine(self) -> None:
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1", tokens_in=None, tokens_out=None)
        dataset = make_dataset(sample)
        engine = MetricsEngine([TokenEfficiencyMetric()], config=MetricConfig())
        report = engine.run(dataset)
        assert "token_efficiency" in report.metric_details
        assert report.metric_details["token_efficiency"]["sessions_with_tokens"] == 0
