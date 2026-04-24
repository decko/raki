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
            "first_pass_success_rate": 0.58,
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


class TestJudgeConfigSerialization:
    """Tests for serialization of judge configuration fields into the report JSON."""

    def test_engine_serializes_llm_provider_into_config(self) -> None:
        """MetricsEngine must serialize llm_provider into the report config dict."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        config = MetricConfig(llm_provider="anthropic", llm_model="claude-3-5-sonnet")
        engine = MetricsEngine([TokenEfficiencyMetric()], config=config)
        report = engine.run(dataset)
        assert report.config["llm_provider"] == "anthropic"

    def test_engine_serializes_llm_temperature_into_config(self) -> None:
        """MetricsEngine must serialize llm_temperature into the report config dict."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        config = MetricConfig(temperature=0.5)
        engine = MetricsEngine([TokenEfficiencyMetric()], config=config)
        report = engine.run(dataset)
        assert report.config["llm_temperature"] == 0.5

    def test_engine_serializes_llm_max_tokens_into_config(self) -> None:
        """MetricsEngine must serialize llm_max_tokens into the report config dict."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        config = MetricConfig(max_tokens=4096)
        engine = MetricsEngine([TokenEfficiencyMetric()], config=config)
        report = engine.run(dataset)
        assert report.config["llm_max_tokens"] == 4096

    def test_judge_fields_none_when_skip_llm(self) -> None:
        """When skip_llm=True, all judge fields must be None in report config."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        config = MetricConfig(
            llm_provider="vertex-anthropic",
            llm_model="claude-sonnet-4-6",
            temperature=0.5,
            max_tokens=4096,
        )
        engine = MetricsEngine([TokenEfficiencyMetric()], config=config)
        report = engine.run(dataset, skip_llm=True)
        assert report.config["llm_provider"] is None
        assert report.config["llm_model"] is None
        assert report.config["llm_temperature"] is None
        assert report.config["llm_max_tokens"] is None

    def test_judge_config_fields_round_trip_in_json(self, tmp_path: Path) -> None:
        """Judge config fields must survive a write/load JSON round-trip."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        config = MetricConfig(
            llm_provider="vertex-anthropic",
            llm_model="claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=4096,
        )
        engine = MetricsEngine([TokenEfficiencyMetric()], config=config)
        report = engine.run(dataset)
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        raw = json.loads(output_path.read_text())
        assert raw["config"]["llm_provider"] == "vertex-anthropic"
        assert raw["config"]["llm_model"] == "claude-sonnet-4-6"
        assert raw["config"]["llm_temperature"] == 0.0
        assert raw["config"]["llm_max_tokens"] == 4096

    def test_old_report_without_judge_fields_loads_cleanly(self, tmp_path: Path) -> None:
        """Old JSON reports without llm_temperature/llm_max_tokens must load without error."""
        old_report = {
            "run_id": "eval-old",
            "timestamp": "2026-01-01T00:00:00Z",
            "config": {
                "llm_model": "claude-sonnet-4-6",
                "metrics": ["cost_efficiency"],
                "skip_llm": False,
            },
            "aggregate_scores": {"cost_efficiency": 1.5},
            "metric_details": {},
            "sample_results": [],
            "manifest_hash": None,
        }
        output_path = tmp_path / "old-report.json"
        output_path.write_text(json.dumps(old_report))
        loaded = load_json_report(output_path)
        assert loaded.run_id == "eval-old"
        assert loaded.config.get("llm_temperature") is None
        assert loaded.config.get("llm_max_tokens") is None

    def test_default_judge_config_fields_present(self) -> None:
        """Default judge config values must appear in the report config dict."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric
        from raki.metrics.protocol import MetricConfig

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        engine = MetricsEngine([TokenEfficiencyMetric()], config=MetricConfig())
        report = engine.run(dataset)
        assert "llm_provider" in report.config
        assert "llm_model" in report.config
        assert "llm_temperature" in report.config
        assert "llm_max_tokens" in report.config


class TestJsonReportContextSource:
    def test_context_source_included_in_json_output(self, tmp_path: Path) -> None:
        """context_source field should be included in JSON report sample results."""
        sample = make_sample("session-ctx-001")
        sample.context_source = "synthesized"
        metric_result = MetricResult(
            name="faithfulness",
            score=0.85,
            sample_scores={"session-ctx-001": 0.85},
        )
        sample_result = SampleResult(sample=sample, scores=[metric_result])
        report = EvalReport(
            run_id="ctx-source-test",
            config={"adapter": "alcove"},
            aggregate_scores={"faithfulness": 0.85},
            sample_results=[sample_result],
        )
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path, include_sessions=True)
        raw = json.loads(output_path.read_text())
        sample_data = raw["sample_results"][0]["sample"]
        assert sample_data["context_source"] == "synthesized"

    def test_context_source_none_in_json_output(self, tmp_path: Path) -> None:
        """context_source=None should appear as null in JSON output."""
        sample = make_sample("session-no-ctx")
        metric_result = MetricResult(
            name="rework_cycles",
            score=1.0,
            sample_scores={"session-no-ctx": 1.0},
        )
        sample_result = SampleResult(sample=sample, scores=[metric_result])
        report = EvalReport(
            run_id="no-ctx-source-test",
            config={"adapter": "session-schema"},
            aggregate_scores={"rework_cycles": 1.0},
            sample_results=[sample_result],
        )
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path, include_sessions=True)
        raw = json.loads(output_path.read_text())
        sample_data = raw["sample_results"][0]["sample"]
        assert sample_data["context_source"] is None


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
        line = format_metric_line("first_pass_success_rate", 0.85, "(32/38 passed)")
        assert "first_pass_success_rate" in line
        assert "0.85" in line

    def test_red_for_low_score(self) -> None:
        line = format_metric_line("first_pass_success_rate", 0.45, "(17/38 passed)")
        assert "0.45" in line
        assert "[red]" in line

    def test_green_for_high_score(self) -> None:
        line = format_metric_line("first_pass_success_rate", 0.85)
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
        line = format_metric_line("first_pass_success_rate", 0.85, sample_count=38)
        assert "(n=38)" in line

    def test_display_name_used_when_provided(self) -> None:
        line = format_metric_line(
            "first_pass_success_rate",
            0.85,
            display_name="First-Pass Verify Rate",
        )
        assert "First-Pass Verify Rate" in line
        assert "first_pass_success_rate" not in line

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
                "first_pass_success_rate": 0.80,
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

    def test_inferred_suffix_when_context_synthesized(self) -> None:
        """Faithfulness/relevancy with synthesized context should show (inferred) tag."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="inferred-test",
            aggregate_scores={
                "faithfulness": 0.85,
                "answer_relevancy": 0.72,
            },
            metric_details={
                "faithfulness": {"context_source": "synthesized", "samples_scored": 5},
                "answer_relevancy": {"context_source": "synthesized", "samples_scored": 5},
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "inferred" in output.lower()

    def test_no_inferred_suffix_when_context_explicit(self) -> None:
        """Faithfulness/relevancy with explicit context should NOT show (inferred) tag."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="explicit-test",
            aggregate_scores={
                "faithfulness": 0.85,
            },
            metric_details={
                "faithfulness": {"context_source": "explicit", "samples_scored": 5},
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "inferred" not in output.lower()


# --- generate_summary_sentence tests ---


def _make_finding(severity: Literal["critical", "major", "minor"]) -> ReviewFinding:
    """Helper to create a ReviewFinding with a given severity."""
    return ReviewFinding(
        reviewer="test-reviewer",
        severity=severity,
        issue=f"Test {severity} issue",
    )


class TestGenerateSummarySentence:
    def test_includes_success_rate_percentage(self) -> None:
        """Summary sentence should include first-pass success rate as a percentage."""
        report = EvalReport(
            run_id="summary-test",
            aggregate_scores={
                "first_pass_success_rate": 0.91,
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
                "first_pass_success_rate": 0.91,
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
                "first_pass_success_rate": 0.91,
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
                "first_pass_success_rate": 0.80,
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
                "first_pass_success_rate": 0.75,
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
                "first_pass_success_rate": 0.8,
                "token_efficiency": 0.0,
            },
            metric_details={
                "first_pass_success_rate": {"passed": 8, "total": 10},
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

    def test_has_no_data_detects_skipped(self) -> None:
        from raki.report.cli_summary import _has_no_data

        details = {
            "faithfulness": {"skipped": "no samples"},
        }
        assert _has_no_data(details, "faithfulness") is True

    def test_has_no_data_returns_false_when_sessions_present(self) -> None:
        from raki.report.cli_summary import _has_no_data

        details = {
            "token_efficiency": {"sessions_with_tokens": 5, "mean_tokens_per_phase": 1500.0},
        }
        assert _has_no_data(details, "token_efficiency") is False

    def test_has_no_data_returns_false_for_missing_metric(self) -> None:
        from raki.report.cli_summary import _has_no_data

        assert _has_no_data({}, "token_efficiency") is False

    def test_no_data_reason_returns_skipped_message(self) -> None:
        from raki.report.cli_summary import _no_data_reason

        details = {
            "faithfulness": {"skipped": "no samples"},
        }
        assert _no_data_reason(details, "faithfulness") == "no samples"

    def test_no_data_reason_returns_default_for_sessions_with(self) -> None:
        from raki.report.cli_summary import _no_data_reason

        details = {
            "token_efficiency": {"sessions_with_tokens": 0},
        }
        assert _no_data_reason(details, "token_efficiency") == "no data"

    def test_print_summary_shows_skipped_reason_for_ragas_metrics(self) -> None:
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="test-skipped-ragas",
            aggregate_scores={
                "faithfulness": None,
                "context_precision": None,
            },
            metric_details={
                "faithfulness": {"skipped": "no samples"},
                "context_precision": {"skipped": "no ground truth"},
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "N/A" in output
        assert "no samples" in output or "no ground truth" in output

    def test_skipped_metric_aggregate_score_is_none_not_zero(self) -> None:
        """Bug #140: aggregate_scores must use None (JSON null) for skipped metrics, not 0.0.

        When a metric is skipped (e.g. no ground truth), the aggregate score
        should be None so CI consumers can distinguish 'not measured' from 'zero'.
        """
        report = EvalReport(
            run_id="test-null-aggregate",
            aggregate_scores={
                "context_precision": None,
                "context_recall": None,
                "first_pass_success_rate": 0.80,
            },
            metric_details={
                "context_precision": {"skipped": "no ground truth"},
                "context_recall": {"skipped": "no ground truth"},
            },
        )
        assert report.aggregate_scores["context_precision"] is None
        assert report.aggregate_scores["context_recall"] is None
        assert report.aggregate_scores["first_pass_success_rate"] == 0.80

    def test_skipped_metric_serializes_to_json_null(self, tmp_path: Path) -> None:
        """Bug #140: None aggregate scores must serialize to JSON null, not 0.0."""
        report = EvalReport(
            run_id="test-json-null",
            aggregate_scores={
                "context_precision": None,
                "context_recall": None,
                "first_pass_success_rate": 0.80,
            },
            metric_details={
                "context_precision": {"skipped": "no ground truth"},
                "context_recall": {"skipped": "no ground truth"},
            },
        )
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        raw = json.loads(output_path.read_text())
        assert raw["aggregate_scores"]["context_precision"] is None
        assert raw["aggregate_scores"]["context_recall"] is None
        assert raw["aggregate_scores"]["first_pass_success_rate"] == 0.80


# --- Three-tier section headers tests (bug #133) ---


class TestThreeTierSectionHeaders:
    """Bug #133: CLI output must show three separate section headers."""

    def _make_console(self) -> tuple:
        from io import StringIO

        from rich.console import Console

        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        return string_io, test_console

    def test_knowledge_metrics_appear_under_knowledge_quality_heading(self) -> None:
        """Knowledge metrics must appear under 'Knowledge Quality', not 'Operational Health'."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="three-tier-test",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "knowledge_gap_rate": 0.10,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "Knowledge Quality" in output

    def test_knowledge_metrics_not_under_operational_heading(self) -> None:
        """Knowledge metrics must NOT be grouped under Operational Health."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="three-tier-separation",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "knowledge_gap_rate": 0.10,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        # The heading "Operational Health" must appear before "Knowledge Quality"
        op_pos = output.find("Operational Health")
        kq_pos = output.find("Knowledge Quality")
        assert op_pos != -1, "Operational Health heading missing"
        assert kq_pos != -1, "Knowledge Quality heading missing"
        assert op_pos < kq_pos, "Operational Health must precede Knowledge Quality"

    def test_three_tier_all_sections_displayed(self) -> None:
        """When all three metric categories exist, all three headings must appear."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="all-three-tiers",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "knowledge_gap_rate": 0.10,
                "faithfulness": 0.90,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "Operational Health" in output
        assert "Knowledge Quality" in output
        assert "Retrieval Quality" in output

    def test_three_sections_appear_in_order(self) -> None:
        """Three sections must appear in the order: Operational → Knowledge → Retrieval."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="order-test",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "knowledge_gap_rate": 0.10,
                "faithfulness": 0.90,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        op_pos = output.find("Operational Health")
        kq_pos = output.find("Knowledge Quality")
        rq_pos = output.find("Retrieval Quality")
        assert op_pos < kq_pos < rq_pos

    def test_only_operational_metrics_shows_one_heading(self) -> None:
        """When only operational metrics are present, only the Operational Health heading shown."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="op-only",
            aggregate_scores={"first_pass_success_rate": 0.80},
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "Operational Health" in output
        # Knowledge Quality and Retrieval Quality must not appear as bold section headings.
        # Nudge text may mention "Knowledge Quality" so check no knowledge metric is listed.
        assert "knowledge_gap_rate" not in output
        assert "knowledge_miss_rate" not in output
        # No retrieval-tier metric should appear either
        assert "faithfulness" not in output

    def test_only_knowledge_metrics_shows_knowledge_heading(self) -> None:
        """When only knowledge metrics are present, only Knowledge Quality heading is shown."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="kq-only",
            aggregate_scores={"knowledge_gap_rate": 0.10},
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "Knowledge Quality" in output
        assert "Operational Health" not in output
        # Nudge text may mention "Retrieval Quality" — assert no retrieval metric is listed
        assert "faithfulness" not in output
        assert "context_precision" not in output


# --- Progression nudge tests (bug #133) ---


class TestProgressionNudges:
    """Bug #133: Each section should show a nudge toward the next tier when it is absent."""

    def _make_console(self) -> tuple:
        from io import StringIO

        from rich.console import Console

        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        return string_io, test_console

    def test_nudge_shown_after_operational_when_no_knowledge_metrics(self) -> None:
        """When only operational metrics are present, show nudge for --docs-path."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="nudge-op-only",
            aggregate_scores={"first_pass_success_rate": 0.80},
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "--docs-path" in output

    def test_nudge_shown_after_knowledge_when_no_retrieval_metrics(self) -> None:
        """When operational + knowledge metrics are present but no retrieval, show --judge nudge."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="nudge-kq-only",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "knowledge_gap_rate": 0.10,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "--judge" in output

    def test_no_docs_path_nudge_when_knowledge_metrics_present(self) -> None:
        """When knowledge metrics are present, the --docs-path nudge must NOT appear."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="nudge-suppressed",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "knowledge_gap_rate": 0.10,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "--docs-path" not in output

    def test_no_judge_nudge_when_retrieval_metrics_present(self) -> None:
        """When retrieval metrics are present, the --judge nudge must NOT appear."""
        string_io, test_console = self._make_console()
        report = EvalReport(
            run_id="nudge-judge-suppressed",
            aggregate_scores={
                "first_pass_success_rate": 0.80,
                "faithfulness": 0.90,
            },
        )
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "--judge" not in output


class TestNoDataMetricDisplayExtended:
    """Additional no-data / null score tests carried from TestNoDataMetricDisplay."""

    def test_skipped_metric_cli_shows_na_not_null(self) -> None:
        """Bug #140: CLI must show 'N/A' for None scores, not 'null' or 'None'."""
        from io import StringIO

        from rich.console import Console

        report = EvalReport(
            run_id="test-cli-na",
            aggregate_scores={
                "context_precision": None,
                "first_pass_success_rate": 0.80,
            },
            metric_details={
                "context_precision": {"skipped": "no ground truth"},
            },
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=True)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "N/A" in output
        assert "null" not in output.lower().replace("n/a", "")
        assert "None" not in output.replace("N/A", "")

    def test_skipped_metric_round_trips_as_none(self, tmp_path: Path) -> None:
        """Bug #140: None aggregate scores must survive JSON round-trip as None."""
        report = EvalReport(
            run_id="test-roundtrip-null",
            aggregate_scores={
                "context_precision": None,
                "first_pass_success_rate": 0.80,
            },
            metric_details={
                "context_precision": {"skipped": "no ground truth"},
            },
        )
        output_path = tmp_path / "report.json"
        write_json_report(report, output_path)
        loaded = load_json_report(output_path)
        assert loaded.aggregate_scores["context_precision"] is None
        assert loaded.aggregate_scores["first_pass_success_rate"] == 0.80

    def test_engine_propagates_none_score_to_aggregate(self) -> None:
        """Bug #140: MetricsEngine must propagate None scores into aggregate_scores.

        Verifies that the engine's dict comprehension preserves None from MetricResult.score.
        """
        results = [
            MetricResult(name="context_precision", score=None, details={"skipped": "no gt"}),
            MetricResult(name="first_pass_success_rate", score=0.80),
        ]
        aggregate = {result.name: result.score for result in results}
        assert aggregate["context_precision"] is None
        assert aggregate["first_pass_success_rate"] == 0.80

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
