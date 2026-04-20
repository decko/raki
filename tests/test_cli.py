"""Tests for CLI commands: raki run, raki validate, raki adapters, raki report, raki report --diff."""

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from raki.cli import main


class TestCliHelp:
    def test_help_shows_all_commands(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "adapters" in result.output


class TestCliAdapters:
    def test_lists_session_schema_adapter(self):
        runner = CliRunner()
        result = runner.invoke(main, ["adapters"])
        assert result.exit_code == 0
        assert "session-schema" in result.output

    def test_lists_alcove_adapter(self):
        runner = CliRunner()
        result = runner.invoke(main, ["adapters"])
        assert result.exit_code == 0
        assert "alcove" in result.output


class TestCliValidate:
    def test_validate_with_manifest(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(empty_manifest)])
        assert result.exit_code == 0

    def test_validate_shows_session_count(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(empty_manifest)])
        assert "0 sessions loaded" in result.output

    def test_validate_shows_skipped_and_errors(self, empty_manifest):
        # Create a directory that no adapter can detect — it will be skipped
        sessions_dir = empty_manifest.parent / "sessions"
        unknown = sessions_dir / "unknown-dir"
        unknown.mkdir()
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(empty_manifest)])
        assert result.exit_code == 0
        assert "skipped" in result.output.lower()

    def test_validate_missing_manifest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["validate"])
        assert result.exit_code != 0

    def test_validate_detailed_output_with_fixture(self, manifest_with_session):
        """Validate shows sessions found, skipped, errors, partial data."""
        manifest, sessions = manifest_with_session
        # Add an unrecognizable directory
        (sessions / "junk").mkdir()
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest)])
        assert result.exit_code == 0
        assert "1 sessions loaded" in result.output
        assert "skipped" in result.output.lower()

    def test_validate_catches_manifest_load_error(self, tmp_path):
        """Validate exits with code 2 when manifest has invalid content."""
        manifest = tmp_path / "raki.yaml"
        manifest.write_text("invalid: yaml: content: [[[")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest)])
        assert result.exit_code == 2
        assert "error" in result.output.lower()


class TestCliRunNoLlm:
    def test_run_no_llm_produces_json(self, manifest_with_session, tmp_path):
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm", "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_run_no_llm_shows_summary(self, manifest_with_session):
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm"],
        )
        assert result.exit_code == 0
        assert "Operational Health" in result.output


class TestCliRunQuiet:
    def test_quiet_mode_suppresses_summary(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "-q"],
        )
        assert result.exit_code == 0
        assert "Operational Health" not in result.output


class TestCliRunDefaultManifest:
    def test_discovers_raki_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest = tmp_path / "raki.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--no-llm"])
        assert result.exit_code == 0

    def test_discovers_eval_manifest_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest = tmp_path / "eval-manifest.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--no-llm"])
        assert result.exit_code == 0

    def test_prints_discovered_manifest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest = tmp_path / "raki.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--no-llm"])
        assert "raki.yaml" in result.output

    def test_error_when_no_manifest_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--no-llm"])
        assert result.exit_code == 2


class TestCliRunThreshold:
    def test_warns_threshold_with_no_llm(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "--threshold", "0.8"],
        )
        assert "threshold" in result.output.lower()
        assert "no-llm" in result.output.lower() or "no_llm" in result.output.lower()

    def test_threshold_exit_code_1_when_below(self, manifest_with_session, tmp_path, monkeypatch):
        """--threshold 0.99 with low retrieval scores should produce exit code 1."""
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"

        # Patch MetricsEngine.run to return a report with low retrieval scores
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={
                "first_pass_verify_rate": 1.0,
                "context_precision": 0.50,
                "context_recall": 0.40,
            },
        )

        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--no-llm",
                    "--threshold",
                    "0.99",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 1


class TestCliRunJson:
    def test_json_stdout_flag(self, manifest_with_session):
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm", "--json", "-q"],
        )
        assert result.exit_code == 0
        # Output should contain valid JSON with run_id
        assert '"run_id"' in result.output

    def test_json_stdout_strips_session_data(self, manifest_with_session):
        """JSON stdout should strip session data by default like write_json_report."""
        import json

        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm", "--json", "-q"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for sample_result in data.get("sample_results", []):
            sample = sample_result.get("sample", {})
            for phase in sample.get("phases", []):
                assert phase["output"] == "<stripped>"

    def test_json_stdout_without_quiet_produces_valid_json(self, manifest_with_session):
        """--json without -q must still produce valid JSON on stdout.

        Rich console output must not corrupt JSON. When --json is set,
        console output should be suppressed or redirected to stderr.
        """
        import json
        import subprocess
        import sys

        manifest, _sessions = manifest_with_session
        proc = subprocess.run(
            [sys.executable, "-m", "raki", "run", "-m", str(manifest), "--no-llm", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0
        # stdout must be parseable JSON — no Rich markup mixed in
        data = json.loads(proc.stdout)
        assert "run_id" in data

    def test_json_flag_sends_rich_output_to_stderr(self, manifest_with_session):
        """When --json is active, Rich console output goes to stderr, not stdout."""
        import subprocess
        import sys

        manifest, _sessions = manifest_with_session
        proc = subprocess.run(
            [sys.executable, "-m", "raki", "run", "-m", str(manifest), "--no-llm", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0
        # stderr should contain human-readable messages (loading, summary, etc.)
        assert len(proc.stderr) > 0, "Expected Rich output on stderr when --json is active"

    def test_json_pipe_roundtrip(self, manifest_with_session):
        """Simulate `raki run --json -m manifest --no-llm | python -m json.tool`."""
        import subprocess
        import sys

        manifest, _sessions = manifest_with_session
        # First get JSON output
        raki_proc = subprocess.run(
            [sys.executable, "-m", "raki", "run", "-m", str(manifest), "--no-llm", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert raki_proc.returncode == 0
        # Pipe through python -m json.tool to validate
        json_tool_proc = subprocess.run(
            [sys.executable, "-m", "json.tool"],
            input=raki_proc.stdout,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert json_tool_proc.returncode == 0, (
            f"json.tool failed: {json_tool_proc.stderr}\nstdout was: {raki_proc.stdout[:500]}"
        )


class TestCliRunVerbose:
    def test_verbose_shows_debug_info(self, tmp_path):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        # Add an unrecognizable directory to trigger skip output
        (sessions / "unknown").mkdir()
        manifest = tmp_path / "raki.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm", "-v"],
        )
        assert result.exit_code == 0
        assert "Skipped" in result.output or "skipped" in result.output.lower()


class TestCliExitCodes:
    def test_exit_code_0_on_success(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm"],
        )
        assert result.exit_code == 0

    def test_exit_code_2_on_error(self, tmp_path):
        """Exit code 2 when manifest not found."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(tmp_path / "nonexistent.yaml")],
        )
        assert result.exit_code == 2


class TestCliUnimplementedOptions:
    def test_warns_adapter_option(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "--adapter", "custom"],
        )
        assert "Warning: --adapter is not yet implemented" in result.output

    def test_warns_metrics_option(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "--metrics", "f1,recall"],
        )
        assert "Warning: --metrics is not yet implemented" in result.output

    def test_warns_tenant_option(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "--tenant", "acme"],
        )
        assert "Warning: --tenant is not yet implemented" in result.output


class TestCliJudgeModel:
    def test_judge_model_option_accepted(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--no-llm",
                "--judge-model",
                "claude-opus-4-6",
            ],
        )
        assert result.exit_code == 0

    def test_judge_model_default(self, empty_manifest):
        """Default judge model should be accepted without errors."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm"],
        )
        assert result.exit_code == 0


class TestCliParallelWiring:
    def test_parallel_option_accepted(self, empty_manifest):
        """--parallel should be accepted without 'not yet implemented' warning."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "-p", "8"],
        )
        assert result.exit_code == 0
        assert "Warning: --parallel" not in result.output


class TestCliValidateQuiet:
    def test_validate_quiet_suppresses_auto_discovery(self, tmp_path, monkeypatch):
        """validate --quiet should suppress auto-discovery messages."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest = tmp_path / "raki.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-q"])
        assert result.exit_code == 0
        assert "Auto-discovered" not in result.output


class TestCliAdaptersDescriptions:
    def test_adapters_shows_description(self):
        """adapters command should show a description for each adapter."""
        runner = CliRunner()
        result = runner.invoke(main, ["adapters"])
        assert result.exit_code == 0
        # Should include adapter descriptions, not just names
        assert "meta.json" in result.output or "events.jsonl" in result.output
        assert "session_id" in result.output or "transcript" in result.output

    def test_adapters_shows_format(self):
        """adapters command should show format type for each adapter."""
        runner = CliRunner()
        result = runner.invoke(main, ["adapters"])
        assert result.exit_code == 0
        assert "directory" in result.output.lower() or "file" in result.output.lower()


class TestCliSummaryDisplayName:
    def test_summary_uses_display_name(self, manifest_with_session):
        """CLI summary should show display_name instead of raw metric names."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm"],
        )
        assert result.exit_code == 0
        # Should show human-readable display names, not raw snake_case
        assert "Verify rate" in result.output
        assert "Rework cycles" in result.output
        assert "Cost / session" in result.output

    def test_summary_does_not_show_raw_names(self, manifest_with_session):
        """CLI summary should not show raw snake_case metric names."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm"],
        )
        assert result.exit_code == 0
        # Raw snake_case names should not appear in the summary lines
        assert "first_pass_verify_rate" not in result.output
        assert "rework_cycles" not in result.output
        assert "cost_efficiency" not in result.output


class TestCliSummaryMetricDescription:
    def test_summary_shows_metric_description(self, manifest_with_session):
        """CLI summary should show metric description in parentheses after score."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--no-llm"],
        )
        assert result.exit_code == 0
        # Metric descriptions should appear in parentheses
        assert "% sessions passing verify" in result.output


def _write_report_json(path, *, include_sessions: bool = False) -> None:
    """Write a minimal EvalReport JSON file for testing the report command."""
    from raki.model.report import EvalReport

    report = EvalReport(
        run_id="test-report-001",
        aggregate_scores={
            "first_pass_verify_rate": 0.85,
            "rework_cycles": 0.3,
            "cost_efficiency": 7.50,
        },
    )
    data = report.model_dump(mode="json")
    if include_sessions:
        # Add a sample_result with non-stripped session data to simulate --include-sessions
        data["sample_results"] = [
            {
                "sample": {
                    "session": {
                        "session_id": "session-101",
                        "started_at": "2026-04-10T00:00:00Z",
                        "total_phases": 3,
                        "rework_cycles": 0,
                        "total_cost_usd": 7.50,
                    },
                    "phases": [
                        {
                            "name": "implement",
                            "generation": 1,
                            "status": "completed",
                            "output": "full implementation output here",
                        },
                        {
                            "name": "verify",
                            "generation": 1,
                            "status": "completed",
                            "output": "PASS",
                        },
                    ],
                    "findings": [],
                    "events": [],
                },
                "scores": [
                    {"name": "first_pass_verify_rate", "score": 1.0},
                ],
            }
        ]
    else:
        # Simulate a stripped report (the default output of raki run)
        data["sample_results"] = [
            {
                "sample": {
                    "session": {
                        "session_id": "session-101",
                        "started_at": "2026-04-10T00:00:00Z",
                        "total_phases": 3,
                        "rework_cycles": 0,
                        "total_cost_usd": 7.50,
                    },
                    "phases": [
                        {
                            "name": "implement",
                            "generation": 1,
                            "status": "completed",
                            "output": "<stripped>",
                        },
                        {
                            "name": "verify",
                            "generation": 1,
                            "status": "completed",
                            "output": "<stripped>",
                        },
                    ],
                    "findings": [],
                    "events": [],
                },
                "scores": [
                    {"name": "first_pass_verify_rate", "score": 1.0},
                ],
            }
        ]
    path.write_text(json.dumps(data, indent=2, default=str))


class TestCliReport:
    def test_report_command_shows_in_help(self):
        """The report command should appear in --help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "report" in result.output

    def test_report_renders_cli_summary(self, tmp_path):
        """raki report --input renders CLI summary to terminal."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(report_json)])
        assert result.exit_code == 0
        assert "Operational Health" in result.output

    def test_report_shows_aggregate_scores(self, tmp_path):
        """raki report should display aggregate metric scores."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(report_json)])
        assert result.exit_code == 0
        assert "0.85" in result.output

    @pytest.mark.skipif(
        not importlib.util.find_spec("jinja2"),
        reason="jinja2 not installed",
    )
    def test_report_generates_html(self, tmp_path):
        """raki report --input --html generates an HTML file."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        html_path = tmp_path / "output.html"
        runner = CliRunner()
        result = runner.invoke(
            main, ["report", "--input", str(report_json), "--html", str(html_path)]
        )
        assert result.exit_code == 0
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content.lower()

    def test_report_warns_when_session_data_stripped(self, tmp_path):
        """Warning shown when session data is stripped (default reports)."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=False)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(report_json)])
        assert result.exit_code == 0
        assert "--include-sessions" in result.output

    def test_report_no_warning_when_session_data_present(self, tmp_path):
        """No warning shown when session data is present."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(report_json)])
        assert result.exit_code == 0
        assert "--include-sessions" not in result.output

    def test_report_exit_code_2_for_missing_input(self, tmp_path):
        """Exit code 2 when input file does not exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(tmp_path / "nonexistent.json")])
        assert result.exit_code == 2

    def test_report_short_input_flag(self, tmp_path):
        """raki report -i works as shorthand for --input."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "-i", str(report_json)])
        assert result.exit_code == 0
        assert "Operational Health" in result.output

    def test_report_html_default_alongside_json(self, tmp_path):
        """When --html is not given, no HTML file is generated."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(report_json)])
        assert result.exit_code == 0
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 0

    def test_report_displays_metric_display_names(self, tmp_path):
        """raki report should show human-readable display names from metadata."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(report_json)])
        assert result.exit_code == 0
        # Should use display names from METRIC_METADATA, not raw metric keys
        assert "Verify rate" in result.output


def _write_diff_report_json(
    path: Path,
    run_id: str = "eval-abc123",
    aggregate_scores: dict | None = None,
    include_sessions: bool = False,
    session_ids: list[str] | None = None,
    rework_cycles: int = 0,
    verify_status: str = "completed",
) -> None:
    """Write a minimal EvalReport JSON file for diff testing."""
    scores = aggregate_scores or {
        "first_pass_verify_rate": 0.85,
        "rework_cycles": 0.3,
        "cost_efficiency": 7.50,
    }
    data = {
        "run_id": run_id,
        "timestamp": "2026-04-10T00:00:00Z",
        "aggregate_scores": scores,
        "sample_results": [],
    }
    if include_sessions and session_ids:
        for session_id in session_ids:
            verify_output = "PASS" if verify_status == "completed" else "FAIL"
            data["sample_results"].append(
                {
                    "sample": {
                        "session": {
                            "session_id": session_id,
                            "started_at": "2026-04-10T00:00:00Z",
                            "total_phases": 3,
                            "rework_cycles": rework_cycles,
                            "total_cost_usd": 7.50,
                        },
                        "phases": [
                            {
                                "name": "implement",
                                "generation": 1,
                                "status": "completed",
                                "output": "done",
                            },
                            {
                                "name": "verify",
                                "generation": 1,
                                "status": verify_status,
                                "output": verify_output,
                            },
                        ],
                        "findings": [],
                        "events": [],
                    },
                    "scores": [],
                }
            )
    path.write_text(json.dumps(data, indent=2, default=str))


class TestCliReportDiff:
    def test_diff_produces_cli_summary(self, tmp_path):
        """raki report --diff baseline.json compare.json produces CLI summary."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="eval-baseline",
            aggregate_scores={"first_pass_verify_rate": 0.78, "rework_cycles": 1.2},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-compare",
            aggregate_scores={"first_pass_verify_rate": 0.91, "rework_cycles": 0.4},
        )
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        assert "eval-baseline" in result.output
        assert "eval-compare" in result.output

    def test_diff_shows_coverage_line(self, tmp_path):
        """Diff output shows session coverage line."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="base",
            include_sessions=True,
            session_ids=["session-101", "session-102"],
        )
        _write_diff_report_json(
            compare,
            run_id="comp",
            include_sessions=True,
            session_ids=["session-101", "session-200"],
        )
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        assert "Matched" in result.output

    def test_diff_exit_code_2_for_missing_file(self, tmp_path):
        """Exit code 2 when baseline or compare file is missing."""
        existing = tmp_path / "existing.json"
        _write_diff_report_json(existing)
        runner = CliRunner()
        # Missing baseline
        result = runner.invoke(
            main, ["report", "--diff", str(tmp_path / "nope.json"), str(existing)]
        )
        assert result.exit_code == 2
        # Missing compare
        result = runner.invoke(
            main, ["report", "--diff", str(existing), str(tmp_path / "nope.json")]
        )
        assert result.exit_code == 2

    def test_diff_warns_no_session_data(self, tmp_path):
        """Warning when either report lacks session data."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(baseline, run_id="base")
        _write_diff_report_json(compare, run_id="comp")
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        assert "include-sessions" in result.output.lower() or "unavailable" in result.output.lower()

    def test_diff_shows_direction_indicators(self, tmp_path):
        """Diff shows direction indicators for metric changes."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="base",
            aggregate_scores={"first_pass_verify_rate": 0.78},
        )
        _write_diff_report_json(
            compare,
            run_id="comp",
            aggregate_scores={"first_pass_verify_rate": 0.91},
        )
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        # Should contain an indicator (green or ▲)
        assert "▲" in result.output or "improved" in result.output.lower()

    def test_diff_warning_banner_when_sessions_dropped(self, tmp_path):
        """Warning banner when sessions are dropped or added."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="base",
            include_sessions=True,
            session_ids=["session-101", "session-102", "session-103"],
        )
        _write_diff_report_json(
            compare,
            run_id="comp",
            include_sessions=True,
            session_ids=["session-101"],
        )
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        assert "dropped" in result.output.lower()

    @pytest.mark.skipif(
        not importlib.util.find_spec("jinja2"),
        reason="jinja2 not installed",
    )
    def test_diff_generates_html(self, tmp_path):
        """raki report --diff --html produces an HTML diff report."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        html_out = tmp_path / "diff.html"
        _write_diff_report_json(
            baseline,
            run_id="eval-base",
            aggregate_scores={"first_pass_verify_rate": 0.78},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-comp",
            aggregate_scores={"first_pass_verify_rate": 0.91},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", "--diff", str(baseline), str(compare), "--html", str(html_out)],
        )
        assert result.exit_code == 0
        assert html_out.exists()
        content = html_out.read_text()
        assert "<html" in content.lower()
        assert "eval-base" in content
        assert "eval-comp" in content

    def test_diff_changed_sessions_grouped(self, tmp_path):
        """Changed sessions are grouped by transition type."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="base",
            include_sessions=True,
            session_ids=["session-101"],
            verify_status="failed",
        )
        _write_diff_report_json(
            compare,
            run_id="comp",
            include_sessions=True,
            session_ids=["session-101"],
            verify_status="completed",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        assert "Improvement" in result.output or "improvement" in result.output.lower()
