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


class TestCliRunDefault:
    def test_run_default_produces_json(self, manifest_with_session, tmp_path):
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_run_default_shows_summary(self, manifest_with_session):
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest)],
        )
        assert result.exit_code == 0
        assert "Operational Health" in result.output


class TestCliRunQuiet:
    def test_quiet_mode_suppresses_summary(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "-q"],
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
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 0

    def test_discovers_eval_manifest_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest = tmp_path / "eval-manifest.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 0

    def test_prints_discovered_manifest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest = tmp_path / "raki.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert "raki.yaml" in result.output

    def test_error_when_no_manifest_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 2


class TestCliRunThreshold:
    def test_warns_threshold_with_no_llm(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--threshold", "0.8"],
        )
        assert "No retrieval metrics active" in result.output

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
                "first_pass_success_rate": 1.0,
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
            ["run", "-m", str(manifest), "--json", "-q"],
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
            ["run", "-m", str(manifest), "--json", "-q"],
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
            [sys.executable, "-m", "raki", "run", "-m", str(manifest), "--json"],
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
            [sys.executable, "-m", "raki", "run", "-m", str(manifest), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0
        # stderr should contain human-readable messages (loading, summary, etc.)
        assert len(proc.stderr) > 0, "Expected Rich output on stderr when --json is active"

    def test_json_pipe_roundtrip(self, manifest_with_session):
        """Simulate `raki run --json -m manifest | python -m json.tool`."""
        import subprocess
        import sys

        manifest, _sessions = manifest_with_session
        # First get JSON output
        raki_proc = subprocess.run(
            [sys.executable, "-m", "raki", "run", "-m", str(manifest), "--json"],
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
            ["run", "-m", str(manifest), "-v"],
        )
        assert result.exit_code == 0
        assert "Skipped" in result.output or "skipped" in result.output.lower()


class TestCliExitCodes:
    def test_exit_code_0_on_success(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest)],
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


class TestAdapterFiltering:
    def test_valid_adapter_name(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--adapter", "session-schema", "-q"],
        )
        assert result.exit_code == 0

    def test_invalid_adapter_name_exits_2(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--adapter", "nonexistent"],
        )
        assert result.exit_code == 2

    def test_invalid_adapter_name_shows_valid_names(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--adapter", "nonexistent"],
        )
        assert "session-schema" in result.output
        assert "alcove" in result.output

    def test_adapter_name_passed_to_loader(self, manifest_with_session):
        """Valid adapter name should be passed through to the loader without warning."""
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--adapter", "session-schema"],
        )
        assert result.exit_code == 0
        assert "Warning: --adapter is not yet implemented" not in result.output


class TestMetricsFiltering:
    def test_filter_single_metric(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--metrics", "cost_efficiency", "-q"],
        )
        assert result.exit_code == 0

    def test_filter_multiple_metrics(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest_path),
                "--metrics",
                "cost_efficiency,rework_cycles",
                "-q",
            ],
        )
        assert result.exit_code == 0

    def test_invalid_metric_name_exits_2(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--metrics", "nonexistent"],
        )
        assert result.exit_code == 2

    def test_invalid_metric_name_shows_valid_names(self, manifest_with_session):
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_path), "--metrics", "nonexistent"],
        )
        assert "Valid metrics" in result.output

    def test_filter_only_runs_selected_metrics(self, manifest_with_session, tmp_path):
        """Filtering to a single metric should only include that metric in the report."""
        manifest_path, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest_path),
                "--metrics",
                "cost_efficiency",
                "-o",
                str(output_dir),
                "--json",
                "-q",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        metric_names = set(data.get("aggregate_scores", {}).keys())
        assert "cost_efficiency" in metric_names
        # Other operational metrics should not appear
        assert "first_pass_success_rate" not in metric_names

    def test_filter_avoids_llm_import_when_only_operational(self, manifest_with_session):
        """When --metrics selects only operational metrics, LLM imports should not happen."""
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest_path),
                "--metrics",
                "cost_efficiency",
                "-q",
            ],
        )
        # Should succeed without LLM setup (default behavior skips LLM)
        assert result.exit_code == 0

    def test_filter_with_spaces_around_names(self, manifest_with_session):
        """Metric names with spaces around commas should be trimmed."""
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest_path),
                "--metrics",
                " cost_efficiency , rework_cycles ",
                "-q",
            ],
        )
        assert result.exit_code == 0

    def test_combined_operational_metrics_default(self, manifest_with_session):
        """--metrics with only operational names should work with default (no LLM)."""
        manifest_path, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest_path),
                "--metrics",
                "first_pass_success_rate,rework_cycles",
                "-q",
            ],
        )
        assert result.exit_code == 0


class TestMetricsSubcommand:
    def test_metrics_command_shows_in_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "metrics" in result.output

    def test_metrics_lists_all(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        # Rich table may truncate long names; check for a shorter metric name
        assert "rework_cycles" in result.output

    def test_metrics_shows_operational_metrics(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "cost_efficiency" in result.output
        assert "rework_cycles" in result.output

    def test_metrics_shows_ragas_metrics(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "context_precision" in result.output
        assert "faithfulness" in result.output

    def test_metrics_shows_display_name(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        # Rich table wraps long display names; check for the un-wrapped prefix
        assert "First-pass success" in result.output

    def test_metrics_shows_requires_llm_column(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "Requires LLM" in result.output

    def test_metrics_shows_higher_is_better_column(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "Higher is Better" in result.output

    def test_metrics_json_output(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "metrics" in data
        metric_names = {metric_info["name"] for metric_info in data["metrics"]}
        assert "first_pass_success_rate" in metric_names
        assert "context_precision" in metric_names

    def test_metrics_json_includes_all_fields(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for metric_info in data["metrics"]:
            assert "name" in metric_info
            assert "display_name" in metric_info
            assert "requires_llm" in metric_info
            assert "higher_is_better" in metric_info

    def test_metrics_json_types(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for metric_info in data["metrics"]:
            assert isinstance(metric_info["name"], str)
            assert isinstance(metric_info["display_name"], str)
            assert isinstance(metric_info["requires_llm"], bool)
            assert isinstance(metric_info["higher_is_better"], bool)

    def test_metrics_shows_knowledge_metrics(self):
        """raki metrics should list knowledge metrics (gap rate and miss rate)."""
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "knowledge_gap_rate" in result.output
        assert "knowledge_miss_rate" in result.output

    def test_metrics_json_includes_knowledge_metrics(self):
        """raki metrics --json should include knowledge metrics in the output."""
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        metric_names = {metric_info["name"] for metric_info in data["metrics"]}
        assert "knowledge_gap_rate" in metric_names
        assert "knowledge_miss_rate" in metric_names

    def test_metrics_knowledge_display_names(self):
        """raki metrics should show human-readable display names for knowledge metrics."""
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        knowledge_metrics = {
            metric_info["name"]: metric_info
            for metric_info in data["metrics"]
            if metric_info["name"] in ("knowledge_gap_rate", "knowledge_miss_rate")
        }
        assert knowledge_metrics["knowledge_gap_rate"]["display_name"] == "Knowledge gap rate"
        assert knowledge_metrics["knowledge_miss_rate"]["display_name"] == "Knowledge miss rate"

    def test_metrics_knowledge_requires_llm_false(self):
        """Knowledge metrics should report requires_llm=False."""
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for metric_info in data["metrics"]:
            if metric_info["name"] in ("knowledge_gap_rate", "knowledge_miss_rate"):
                assert metric_info["requires_llm"] is False

    def test_metrics_knowledge_higher_is_better_false(self):
        """Knowledge metrics (gap/miss rate) should report higher_is_better=False."""
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for metric_info in data["metrics"]:
            if metric_info["name"] in ("knowledge_gap_rate", "knowledge_miss_rate"):
                assert metric_info["higher_is_better"] is False


class TestCliJudgeModel:
    def test_judge_model_option_accepted(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
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
            ["run", "-m", str(empty_manifest)],
        )
        assert result.exit_code == 0


class TestCliJudgeProvider:
    def test_judge_provider_vertex_anthropic_accepted(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--judge-provider",
                "vertex-anthropic",
            ],
        )
        assert result.exit_code == 0

    def test_judge_provider_anthropic_accepted(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--judge-provider",
                "anthropic",
            ],
        )
        assert result.exit_code == 0

    def test_judge_provider_invalid_rejected(self, empty_manifest):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--judge-provider",
                "openai",
            ],
        )
        assert result.exit_code == 2

    def test_judge_provider_default(self, empty_manifest):
        """Default judge provider (vertex-anthropic) should be accepted without errors."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest)],
        )
        assert result.exit_code == 0


class TestCliParallelWiring:
    def test_parallel_option_accepted(self, empty_manifest):
        """--parallel should be accepted without 'not yet implemented' warning."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "-p", "8"],
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
            ["run", "-m", str(manifest)],
        )
        assert result.exit_code == 0
        # Should show human-readable display names, not raw snake_case
        assert "First-pass success rate" in result.output
        assert "Rework cycles" in result.output
        assert "Cost / session" in result.output

    def test_summary_does_not_show_raw_names(self, manifest_with_session):
        """CLI summary should not show raw snake_case metric names."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest)],
        )
        assert result.exit_code == 0
        # Raw snake_case names should not appear in the summary lines
        assert "first_pass_success_rate" not in result.output
        assert "rework_cycles" not in result.output
        assert "cost_efficiency" not in result.output


class TestCliSummaryMetricDescription:
    def test_summary_shows_metric_description(self, manifest_with_session):
        """CLI summary should show metric description in parentheses after score."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest)],
        )
        assert result.exit_code == 0
        # Metric descriptions should appear in parentheses
        assert "% sessions with no rework" in result.output


def _write_report_json(path, *, include_sessions: bool = False) -> None:
    """Write a minimal EvalReport JSON file for testing the report command."""
    from raki.model.report import EvalReport

    report = EvalReport(
        run_id="test-report-001",
        aggregate_scores={
            "first_pass_success_rate": 0.85,
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
                    {"name": "first_pass_success_rate", "score": 1.0},
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
                    {"name": "first_pass_success_rate", "score": 1.0},
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
        """raki report file.json renders CLI summary to terminal."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "Operational Health" in result.output

    def test_report_shows_aggregate_scores(self, tmp_path):
        """raki report should display aggregate metric scores."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "0.85" in result.output

    @pytest.mark.skipif(
        not importlib.util.find_spec("jinja2"),
        reason="jinja2 not installed",
    )
    def test_report_generates_html(self, tmp_path):
        """raki report file.json --html generates an HTML file."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        html_path = tmp_path / "output.html"
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json), "--html", str(html_path)])
        assert result.exit_code == 0
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content.lower()

    def test_report_warns_when_session_data_stripped(self, tmp_path):
        """Warning shown when session data is stripped (default reports)."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=False)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "--include-sessions" in result.output

    def test_report_no_warning_when_session_data_present(self, tmp_path):
        """No warning shown when session data is present."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "--include-sessions" not in result.output

    def test_report_exit_code_2_for_missing_input(self, tmp_path):
        """Exit code 2 when input file does not exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(tmp_path / "nonexistent.json")])
        assert result.exit_code == 2

    def test_report_html_default_alongside_json(self, tmp_path):
        """When --html is not given, no HTML file is generated."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 0

    def test_report_displays_metric_display_names(self, tmp_path):
        """raki report should show human-readable display names from metadata."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        # Should use display names from METRIC_METADATA, not raw metric keys
        assert "First-pass success rate" in result.output


def _write_diff_report_json(
    path: Path,
    run_id: str = "eval-abc123",
    aggregate_scores: dict | None = None,
    include_sessions: bool = False,
    session_ids: list[str] | None = None,
    rework_cycles: int = 0,
    verify_status: str = "completed",
    config: dict | None = None,
) -> None:
    """Write a minimal EvalReport JSON file for diff testing."""
    scores = aggregate_scores or {
        "first_pass_success_rate": 0.85,
        "rework_cycles": 0.3,
        "cost_efficiency": 7.50,
    }
    data: dict = {
        "run_id": run_id,
        "timestamp": "2026-04-10T00:00:00Z",
        "aggregate_scores": scores,
        "sample_results": [],
    }
    if config is not None:
        data["config"] = config
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


class TestGroundTruthWiring:
    def test_run_loads_ground_truth_when_configured(self, manifest_with_ground_truth):
        """run() should succeed when manifest has ground_truth.path set."""
        manifest_path, _sessions, _gt = manifest_with_ground_truth
        runner = CliRunner()
        result = runner.invoke(main, ["run", "-m", str(manifest_path), "-q"])
        assert result.exit_code == 0

    def test_run_logs_match_count(self, manifest_with_ground_truth):
        """run() should log how many sessions matched ground truth."""
        manifest_path, _sessions, _gt = manifest_with_ground_truth
        runner = CliRunner()
        result = runner.invoke(main, ["run", "-m", str(manifest_path)])
        assert "Matched ground truth" in result.output

    def test_run_warns_low_match_rate(self, manifest_with_ground_truth):
        """run() should warn when match rate is below 50%."""
        manifest_path, _sessions, _gt = manifest_with_ground_truth
        runner = CliRunner()
        result = runner.invoke(main, ["run", "-m", str(manifest_path)])
        # The pass-simple fixture has no triage phase with code_area, so 0/1 match rate
        assert (
            "ground truth matching relies on" in result.output
            or "Matched ground truth for 0" in result.output
        )

    def test_run_graceful_on_corrupt_ground_truth(self, tmp_path, pass_simple_dir):
        """run() should continue gracefully when ground truth YAML is corrupt."""
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        session_dest = sessions / "101"
        session_dest.mkdir()
        for file_path in pass_simple_dir.iterdir():
            (session_dest / file_path.name).write_text(file_path.read_text())

        ground_truth = tmp_path / "curated.yaml"
        ground_truth.write_text("not: valid: yaml: [[[")

        manifest = tmp_path / "raki.yaml"
        manifest.write_text(
            f"sessions:\n  path: {sessions}\n  format: auto\n"
            f"ground_truth:\n  path: {ground_truth}\n"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["run", "-m", str(manifest)])
        assert result.exit_code == 0

    def test_run_quiet_suppresses_match_log(self, manifest_with_ground_truth):
        """run() in quiet mode should not log match count."""
        manifest_path, _sessions, _gt = manifest_with_ground_truth
        runner = CliRunner()
        result = runner.invoke(main, ["run", "-m", str(manifest_path), "-q"])
        assert result.exit_code == 0
        assert "Matched ground truth" not in result.output

    def test_validate_reports_ground_truth_status(self, manifest_with_ground_truth):
        """validate command should report ground truth status when configured."""
        manifest_path, _sessions, _gt = manifest_with_ground_truth
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest_path)])
        assert "ground truth" in result.output.lower()

    def test_validate_graceful_on_corrupt_ground_truth(self, tmp_path, pass_simple_dir):
        """validate command should warn on corrupt ground truth YAML."""
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        session_dest = sessions / "101"
        session_dest.mkdir()
        for file_path in pass_simple_dir.iterdir():
            (session_dest / file_path.name).write_text(file_path.read_text())

        ground_truth = tmp_path / "curated.yaml"
        ground_truth.write_text("not: valid: yaml: [[[")

        manifest = tmp_path / "raki.yaml"
        manifest.write_text(
            f"sessions:\n  path: {sessions}\n  format: auto\n"
            f"ground_truth:\n  path: {ground_truth}\n"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest)])
        assert result.exit_code == 0

    def test_run_with_matching_ground_truth(self, tmp_path, pass_simple_dir):
        """run() should match ground truth when session has matching triage code_area."""
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        session_dest = sessions / "101"
        session_dest.mkdir()
        for file_path in pass_simple_dir.iterdir():
            (session_dest / file_path.name).write_text(file_path.read_text())
        # Patch triage.json to include code_area for domain matching
        triage_path = session_dest / "triage.json"
        triage_path.write_text(
            '{"ticket_key": "101", "complexity": "small", '
            '"approach": "Add pydantic validation", '
            '"automatable": true, "code_area": "api validation"}'
        )

        ground_truth = tmp_path / "curated.yaml"
        ground_truth.write_text(
            "- question: How does API validation work?\n"
            "  expected_approach: Use pydantic\n"
            "  domains:\n"
            "    - api\n"
            "    - validation\n"
        )

        manifest = tmp_path / "raki.yaml"
        manifest.write_text(
            f"sessions:\n  path: {sessions}\n  format: auto\n"
            f"ground_truth:\n  path: {ground_truth}\n"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["run", "-m", str(manifest)])
        assert result.exit_code == 0
        assert "Matched ground truth for 1/1" in result.output


class TestCliReportDiff:
    def test_diff_produces_cli_summary(self, tmp_path):
        """raki report --diff baseline.json compare.json produces CLI summary."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="eval-baseline",
            aggregate_scores={"first_pass_success_rate": 0.78, "rework_cycles": 1.2},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-compare",
            aggregate_scores={"first_pass_success_rate": 0.91, "rework_cycles": 0.4},
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
            aggregate_scores={"first_pass_success_rate": 0.78},
        )
        _write_diff_report_json(
            compare,
            run_id="comp",
            aggregate_scores={"first_pass_success_rate": 0.91},
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
            aggregate_scores={"first_pass_success_rate": 0.78},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-comp",
            aggregate_scores={"first_pass_success_rate": 0.91},
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

    def test_diff_exit_code_0_with_judge_config_warnings(self, tmp_path):
        """Judge config warnings should NOT affect exit code (still 0)."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="base",
            aggregate_scores={"first_pass_success_rate": 0.78},
            config={
                "skip_llm": False,
                "llm_provider": "anthropic",
                "llm_model": "claude-opus-4",
            },
        )
        _write_diff_report_json(
            compare,
            run_id="comp",
            aggregate_scores={"first_pass_success_rate": 0.91},
            config={
                "skip_llm": False,
                "llm_provider": "anthropic",
                "llm_model": "claude-sonnet-4-6",
            },
        )
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--diff", str(baseline), str(compare)])
        assert result.exit_code == 0
        assert "claude-opus-4" in result.output
        assert "claude-sonnet-4-6" in result.output


class TestCLIInversion:
    """Tests for issue #112: --judge opt-in, --no-llm deprecated."""

    def test_run_defaults_to_no_llm(self, manifest_with_session, tmp_path):
        """run without --judge should NOT run LLM metrics (skip_llm=True)."""
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir), "-q"],
        )
        assert result.exit_code == 0
        # Should produce only operational metrics, not LLM metrics
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "faithfulness" not in data.get("aggregate_scores", {})

    def test_run_with_judge_enables_llm(self, empty_manifest):
        """run --judge should set skip_llm=False, enabling LLM metric imports."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"faithfulness": 0.9},
        )
        runner = CliRunner()
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report) as mock_run:
            result = runner.invoke(
                main,
                ["run", "-m", str(empty_manifest), "--judge", "-q"],
            )
            assert result.exit_code == 0
            # Engine.run should have been called with skip_llm=False
            mock_run.assert_called_once()
            _call_args, call_kwargs = mock_run.call_args
            assert call_kwargs.get("skip_llm") is False

    def test_no_llm_prints_deprecation(self, empty_manifest):
        """--no-llm should print a deprecation warning to stderr."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--no-llm", "-q"],
        )
        assert result.exit_code == 0
        assert "--no-llm is deprecated" in result.stderr

    def test_judge_and_no_llm_conflict(self, empty_manifest):
        """--judge --no-llm should produce an error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--judge", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "--judge" in result.output and "--no-llm" in result.output

    def test_report_defaults_to_no_llm(self, manifest_with_session, tmp_path):
        """report subcommand is not affected (report doesn't have --no-llm/--judge)."""
        # The report command re-renders from saved JSON, it doesn't run metrics.
        # This test verifies report still works without any new flags.
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "Operational Health" in result.output

    def test_report_with_judge_flag_is_not_accepted(self):
        """report subcommand should NOT accept --judge (it doesn't run metrics)."""
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--judge"])
        assert result.exit_code == 2


class TestTenantRemoved:
    def test_tenant_option_no_longer_exists(self):
        """--tenant was removed in v0.6.0 and should produce a Click error."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--tenant", "foo"])
        assert result.exit_code == 2

    def test_tenant_option_shows_no_such_option(self):
        """Click should report --tenant as an unrecognized option."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--tenant", "foo"])
        assert "No such option" in result.output or "no such option" in result.output


class TestThresholdWarningUpdated:
    def test_threshold_without_judge_warns_no_retrieval_metrics_active(self, empty_manifest):
        """--threshold without --judge should warn about no retrieval metrics being active."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--threshold", "0.5"],
        )
        assert "No retrieval metrics active" in result.output

    def test_threshold_warning_mentions_operational_scales(self, empty_manifest):
        """Warning should explain that operational metrics use non-0-1 scales."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--threshold", "0.5"],
        )
        assert "non-0-1 scales" in result.output

    def test_threshold_warning_mentions_v070(self, empty_manifest):
        """Warning should mention per-metric thresholds planned for v0.7.0."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(empty_manifest), "--threshold", "0.5"],
        )
        assert "v0.7.0" in result.output


class TestReportPositionalArg:
    def test_report_accepts_positional_path(self, tmp_path):
        """raki report file.json should work (positional argument)."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "Operational Health" in result.output

    def test_report_positional_shows_aggregate_scores(self, tmp_path):
        """Positional report path should display aggregate scores."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_json)])
        assert result.exit_code == 0
        assert "0.85" in result.output

    def test_report_without_path_or_diff_errors(self):
        """raki report with no arguments should error with exit code 2."""
        runner = CliRunner()
        result = runner.invoke(main, ["report"])
        assert result.exit_code == 2

    def test_report_missing_positional_file_errors(self, tmp_path):
        """Positional path to nonexistent file should produce exit code 2."""
        runner = CliRunner()
        result = runner.invoke(main, ["report", str(tmp_path / "nonexistent.json")])
        assert result.exit_code == 2


class TestDocsPathCLI:
    """Tests for --docs-path flag on the run command."""

    def test_docs_path_option_accepted(self, manifest_with_session, tmp_path, monkeypatch):
        """--docs-path should be accepted as a valid CLI option."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide\nSome documentation.\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--docs-path", str(docs_dir), "-q"],
        )
        assert result.exit_code == 0

    def test_docs_path_nonexistent_exits_2(self, manifest_with_session, tmp_path):
        """--docs-path pointing to nonexistent dir should fail."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--docs-path", str(tmp_path / "nope")],
        )
        assert result.exit_code == 2

    def test_docs_path_adds_knowledge_metrics(self, manifest_with_session, tmp_path, monkeypatch):
        """--docs-path should add knowledge metrics to the output."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide\nSome documentation content.\n")
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest),
                "--docs-path",
                str(docs_dir),
                "-o",
                str(output_dir),
                "--json",
                "-q",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        metric_names = set(data.get("aggregate_scores", {}).keys())
        assert "knowledge_gap_rate" in metric_names
        assert "knowledge_miss_rate" in metric_names

    def test_docs_path_logs_loaded_count(self, manifest_with_session, tmp_path, monkeypatch):
        """--docs-path should log how many chunks were loaded."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide\nContent.\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "--docs-path", str(docs_dir)],
        )
        assert result.exit_code == 0
        assert "doc" in result.output.lower()

    def test_docs_path_overrides_manifest_docs(self, tmp_path, pass_simple_dir, monkeypatch):
        """CLI --docs-path should override manifest docs.path."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        session_dest = sessions / "101"
        session_dest.mkdir()
        for file_path in pass_simple_dir.iterdir():
            (session_dest / file_path.name).write_text(file_path.read_text())

        # Manifest points to one docs dir
        manifest_docs = tmp_path / "manifest_docs"
        manifest_docs.mkdir()
        (manifest_docs / "old.md").write_text("# Old docs\n")

        manifest_file = tmp_path / "raki.yaml"
        manifest_file.write_text(f"sessions:\n  path: {sessions}\ndocs:\n  path: {manifest_docs}\n")

        # CLI overrides to a different docs dir
        cli_docs = tmp_path / "cli_docs"
        cli_docs.mkdir()
        (cli_docs / "new.md").write_text("# New docs\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_file), "--docs-path", str(cli_docs), "-q"],
        )
        assert result.exit_code == 0

    def test_manifest_docs_used_when_no_cli_flag(self, tmp_path, pass_simple_dir, monkeypatch):
        """Manifest docs.path should be used when --docs-path is not given."""
        monkeypatch.chdir(tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        session_dest = sessions / "101"
        session_dest.mkdir()
        for file_path in pass_simple_dir.iterdir():
            (session_dest / file_path.name).write_text(file_path.read_text())

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide\nContent here.\n")

        manifest_file = tmp_path / "raki.yaml"
        manifest_file.write_text(f"sessions:\n  path: {sessions}\ndocs:\n  path: {docs_dir}\n")

        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest_file),
                "-o",
                str(output_dir),
                "--json",
                "-q",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        metric_names = set(data.get("aggregate_scores", {}).keys())
        assert "knowledge_gap_rate" in metric_names

    def test_docs_path_outside_project_root_rejected(self, tmp_path, monkeypatch):
        """--docs-path pointing outside the CWD (project root) should fail with UsageError."""
        # Create project directory — this is the CWD (project root)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        sessions = project_dir / "sessions"
        sessions.mkdir()
        manifest_file = project_dir / "raki.yaml"
        manifest_file.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")

        # Create docs directory outside the CWD (project root)
        outside_docs = tmp_path / "outside_docs"
        outside_docs.mkdir()
        (outside_docs / "guide.md").write_text("# Guide\nContent.\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_file), "--docs-path", str(outside_docs), "-q"],
        )
        assert result.exit_code != 0
        assert "must be within the project root" in result.output

    def test_docs_path_within_cwd_but_outside_manifest_parent_accepted(self, tmp_path, monkeypatch):
        """--docs-path within CWD but outside manifest parent directory should be accepted.

        Regression test for bug where the guard used manifest_file.parent as the
        project root instead of CWD.  When the manifest lives in a subdirectory
        (e.g. discovered from a nested config/ folder), docs at the repo root must
        still be accepted.
        """
        # CWD is the project root (tmp_path)
        monkeypatch.chdir(tmp_path)

        # Manifest is in a subdirectory — simulates auto-discovery from a nested location
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        sessions = config_dir / "sessions"
        sessions.mkdir()
        manifest_file = config_dir / "raki.yaml"
        manifest_file.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")

        # docs is at the project root — within CWD but NOT within manifest parent (config/)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide\nContent.\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest_file), "--docs-path", str(docs_dir), "-q"],
        )
        # Should succeed: docs_dir is within CWD even though it's outside config/
        assert result.exit_code == 0


class TestRagasMetricsSync:
    """_RAGAS_METRICS dict keys must stay in sync with actual Ragas metric class .name attributes."""

    def test_ragas_metrics_keys_match_class_names(self):
        """The set of _RAGAS_METRICS keys must equal the set of Ragas metric .name values."""
        from raki.cli import _RAGAS_METRICS
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric
        from raki.metrics.ragas.precision import ContextPrecisionMetric
        from raki.metrics.ragas.recall import ContextRecallMetric
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        ragas_classes = [
            FaithfulnessMetric,
            ContextPrecisionMetric,
            ContextRecallMetric,
            AnswerRelevancyMetric,
        ]
        class_names = {cls.name for cls in ragas_classes}
        dict_keys = set(_RAGAS_METRICS.keys())
        assert dict_keys == class_names, (
            f"_RAGAS_METRICS keys {dict_keys} do not match "
            f"Ragas metric class .name attributes {class_names}"
        )


class TestValidateDeep:
    """Tests for raki validate --deep flag."""

    def test_deep_flag_accepted(self, manifest_with_session):
        """--deep should be an accepted option on the validate command."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0

    def test_deep_runs_adapter_check(self, manifest_with_session):
        """--deep should check that each adapter can load a session."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        assert "Adapter" in result.output
        assert "session-schema" in result.output

    def test_deep_runs_operational_metrics_check(self, manifest_with_session):
        """--deep should run operational metrics against a single sample."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        assert "Operational metrics" in result.output

    def test_deep_shows_pass_indicators(self, manifest_with_session):
        """--deep should show pass/fail indicators per check."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        # Should contain checkmark or PASS indicators
        assert "\u2713" in result.output

    def test_deep_checks_ground_truth_when_configured(self, manifest_with_ground_truth):
        """--deep should check ground truth loading and matching when configured."""
        manifest_path, _sessions, _gt = manifest_with_ground_truth
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest_path), "--deep"])
        assert result.exit_code == 0
        assert "ground truth" in result.output.lower()

    def test_deep_no_ground_truth_skips_check(self, manifest_with_session):
        """--deep should skip ground truth check when not configured."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        # Should not mention ground truth loading/matching (only the standard validate output)
        # The deep section should not have ground truth checks
        assert "Ground truth loading" not in result.output

    def test_deep_no_llm_calls(self, manifest_with_session):
        """--deep should not trigger any LLM metric computation."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        # Should not mention any LLM/Ragas metrics
        assert "faithfulness" not in result.output.lower()
        assert "context_precision" not in result.output.lower()

    def test_deep_no_report_generation(self, manifest_with_session, tmp_path):
        """--deep should not generate any report files."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        # No report files should be created in the current or results directory
        json_files = list(tmp_path.glob("**/*.json"))
        html_files = list(tmp_path.glob("**/*.html"))
        # Only the manifest yaml should exist, no report outputs
        assert not any("raki-report" in str(filepath) for filepath in json_files)
        assert not any("raki-report" in str(filepath) for filepath in html_files)

    def test_deep_with_empty_sessions_shows_skip(self, empty_manifest):
        """--deep with no sessions should report that adapter checks are skipped."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(empty_manifest), "--deep"])
        assert result.exit_code == 0
        # Should indicate no sessions are available for deep checks
        assert "no sessions" in result.output.lower() or "skip" in result.output.lower()

    def test_deep_shows_metric_results(self, manifest_with_session):
        """--deep should show individual metric results from the single-sample run."""
        manifest, _sessions = manifest_with_session
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        assert result.exit_code == 0
        # Should show at least one metric name or display name
        assert (
            "First-pass success rate" in result.output or "first_pass_success_rate" in result.output
        )

    def test_deep_adapter_failure_shows_fail(self, tmp_path):
        """--deep should show a failure indicator when an adapter fails to load."""
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        # Create a malformed session directory that will fail to load
        bad_session = sessions / "bad-session"
        bad_session.mkdir()
        # session-schema expects meta.json; create one with invalid content
        (bad_session / "meta.json").write_text("not valid json {{{")
        manifest = tmp_path / "raki.yaml"
        manifest.write_text(f"sessions:\n  path: {sessions}\n  format: auto\n")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "-m", str(manifest), "--deep"])
        # Should still exit 0 (deep is informational) but show failure indicators
        assert result.exit_code == 0
        # Should contain a failure indicator
        assert "\u2717" in result.output or "fail" in result.output.lower()


class TestGateThresholdCLI:
    """Tests for --gate per-metric quality gates on the run command."""

    def test_gate_pass(self, manifest_with_session, tmp_path):
        """--gate with a passing threshold should exit 0."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"first_pass_success_rate": 0.90},
        )
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--gate",
                    "first_pass_success_rate>0.80",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 0
            assert "PASS" in result.output

    def test_gate_violation_exits_1(self, manifest_with_session, tmp_path):
        """--gate with a failing threshold should exit 1."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"first_pass_success_rate": 0.70},
        )
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--gate",
                    "first_pass_success_rate>0.80",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 1
            assert "FAIL" in result.output

    def test_gate_na_skip(self, manifest_with_session, tmp_path):
        """--gate with N/A metric should skip and exit 0."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"faithfulness": None},
        )
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--gate",
                    "faithfulness>0.80",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 0
            assert "SKIP" in result.output

    def test_require_metric_fails_on_na(self, manifest_with_session, tmp_path):
        """--require-metric with N/A metric should exit 1."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"faithfulness": None},
        )
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--gate",
                    "faithfulness>0.80",
                    "--require-metric",
                    "faithfulness",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 1
            assert "FAIL" in result.output

    def test_manifest_thresholds_used_when_no_cli_gate(self, tmp_path):
        """Manifest thresholds should be used when no --gate is specified."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        # Create a manifest with thresholds
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text(
            f"sessions:\n  path: {sessions}\n  format: auto\n"
            f"thresholds:\n  - first_pass_success_rate>0.99\n"
        )

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"first_pass_success_rate": 0.80},
        )
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest_path),
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 1
            assert "FAIL" in result.output

    def test_cli_gate_overrides_manifest_thresholds(self, tmp_path):
        """CLI --gate should override manifest thresholds."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        # Manifest threshold is strict (>0.99), but CLI gate is lenient (>0.50)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        manifest_path = tmp_path / "raki.yaml"
        manifest_path.write_text(
            f"sessions:\n  path: {sessions}\n  format: auto\n"
            f"thresholds:\n  - first_pass_success_rate>0.99\n"
        )

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"first_pass_success_rate": 0.80},
        )
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest_path),
                    "--gate",
                    "first_pass_success_rate>0.50",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 0
            assert "PASS" in result.output

    def test_gate_invalid_syntax_exits_2(self, manifest_with_session, tmp_path):
        """--gate with invalid syntax should exit 2."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"first_pass_success_rate": 0.90},
        )
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--gate",
                    "invalid threshold syntax",
                    "-o",
                    str(output_dir),
                ],
            )
            assert result.exit_code == 2

    def test_gate_quiet_suppresses_output(self, manifest_with_session, tmp_path):
        """--gate with -q should suppress quality gates output."""
        from unittest.mock import patch

        from raki.model.report import EvalReport

        fake_report = EvalReport(
            run_id="fake",
            aggregate_scores={"first_pass_success_rate": 0.90},
        )
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        with patch("raki.metrics.MetricsEngine.run", return_value=fake_report):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "run",
                    "-m",
                    str(manifest),
                    "--gate",
                    "first_pass_success_rate>0.80",
                    "-o",
                    str(output_dir),
                    "-q",
                ],
            )
            assert result.exit_code == 0
            assert "Quality Gates" not in result.output

    def test_gate_unknown_metric_exits_2(self, empty_manifest):
        """--gate with a completely unknown metric name should exit 2 (not silently skip)."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--gate",
                "completely_fake_metric>0.5",
            ],
        )
        assert result.exit_code == 2

    def test_gate_unknown_metric_shows_error_message(self, empty_manifest):
        """--gate with an unknown metric name should show a friendly error message."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--gate",
                "completely_fake_metric>0.5",
            ],
        )
        assert "completely_fake_metric" in result.output
        assert "unknown" in result.output.lower() or "invalid" in result.output.lower()

    def test_gate_unknown_metric_lists_valid_metrics(self, empty_manifest):
        """--gate with an unknown metric name should list valid metric names."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--gate",
                "completely_fake_metric>0.5",
            ],
        )
        assert "first_pass_success_rate" in result.output
        assert "faithfulness" in result.output

    def test_gate_known_but_uncomputed_metric_still_skips(self, empty_manifest):
        """--gate with a known-but-not-computed metric (e.g. faithfulness without --judge)
        should SKIP gracefully (exit 0), not exit 2."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--gate",
                "faithfulness>0.85",
                "-q",
            ],
        )
        # faithfulness is a valid metric name, so no error
        assert result.exit_code == 0

    def test_gate_validation_happens_before_evaluation(self, empty_manifest):
        """--gate validation of metric names should happen before heavy evaluation."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(empty_manifest),
                "--gate",
                "typo_metricc>0.5",
            ],
        )
        assert result.exit_code == 2
        # The error message should mention --gate
        assert "--gate" in result.output or "gate" in result.output.lower()


class TestRegressionCLI:
    """Tests for --fail-on-regression on the report --diff command."""

    def test_fail_on_regression_detects_regression(self, tmp_path):
        """--fail-on-regression should exit non-zero when a metric regresses."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="eval-baseline",
            aggregate_scores={"first_pass_success_rate": 0.90},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-compare",
            aggregate_scores={"first_pass_success_rate": 0.70},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "report",
                "--diff",
                str(baseline),
                str(compare),
                "--fail-on-regression",
            ],
        )
        assert result.exit_code == 3
        assert "Regressions detected" in result.output

    def test_fail_on_regression_exits_0_when_no_regression(self, tmp_path):
        """--fail-on-regression should exit 0 when no regression is detected."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="eval-baseline",
            aggregate_scores={"first_pass_success_rate": 0.80},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-compare",
            aggregate_scores={"first_pass_success_rate": 0.90},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "report",
                "--diff",
                str(baseline),
                str(compare),
                "--fail-on-regression",
            ],
        )
        assert result.exit_code == 0

    def test_fail_on_regression_flag_accepted_without_diff(self, tmp_path):
        """--fail-on-regression without --diff should not crash (it's ignored)."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--fail-on-regression"],
        )
        assert result.exit_code == 0

    def test_diff_without_fail_on_regression_exits_0(self, tmp_path):
        """--diff without --fail-on-regression should exit 0 even with regression."""
        baseline = tmp_path / "baseline.json"
        compare = tmp_path / "compare.json"
        _write_diff_report_json(
            baseline,
            run_id="eval-baseline",
            aggregate_scores={"first_pass_success_rate": 0.90},
        )
        _write_diff_report_json(
            compare,
            run_id="eval-compare",
            aggregate_scores={"first_pass_success_rate": 0.70},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", "--diff", str(baseline), str(compare)],
        )
        assert result.exit_code == 0


class TestReportGates:
    """Tests for --gate and --require-metric flags on the report subcommand (ticket #139)."""

    def test_report_gate_option_accepted(self, tmp_path):
        """--gate should be a valid option on the report command."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "first_pass_success_rate>0.50"],
        )
        assert result.exit_code == 0

    def test_report_require_metric_option_accepted(self, tmp_path):
        """--require-metric should be a valid option on the report command."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "report",
                str(report_json),
                "--gate",
                "first_pass_success_rate>0.50",
                "--require-metric",
                "first_pass_success_rate",
            ],
        )
        assert result.exit_code == 0

    def test_report_gate_pass_exits_0(self, tmp_path):
        """--gate with a passing threshold should exit 0 and show PASS."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        # report.json has first_pass_success_rate=0.85, threshold is 0.50 so it passes
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "first_pass_success_rate>0.50"],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_report_gate_violation_exits_1(self, tmp_path):
        """--gate with a failing threshold should exit 1 and show FAIL."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        # report.json has first_pass_success_rate=0.85, threshold >0.99 fails
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "first_pass_success_rate>0.99"],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_report_gate_na_metric_skipped(self, tmp_path):
        """--gate with an N/A metric (not in report) should SKIP and exit 0."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        # faithfulness is not in the report (N/A) — should skip, not fail
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "faithfulness>0.80"],
        )
        assert result.exit_code == 0
        assert "SKIP" in result.output

    def test_report_require_metric_fails_on_na(self, tmp_path):
        """--require-metric with an N/A metric should exit 1 and show FAIL."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        # faithfulness is not in the report (N/A) + required -> FAIL
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "report",
                str(report_json),
                "--gate",
                "faithfulness>0.80",
                "--require-metric",
                "faithfulness",
            ],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_report_gate_invalid_syntax_exits_2(self, tmp_path):
        """--gate with invalid syntax should exit 2."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "not a valid gate"],
        )
        assert result.exit_code == 2

    def test_report_gate_quiet_suppresses_gate_output(self, tmp_path):
        """--gate with -q should suppress Quality Gates output."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "first_pass_success_rate>0.50", "-q"],
        )
        assert result.exit_code == 0
        assert "Quality Gates" not in result.output

    def test_report_gate_multiple_gates(self, tmp_path):
        """Multiple --gate flags should all be evaluated."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        # first_pass_success_rate=0.85 > 0.50 PASS; rework_cycles=0.3 < 1.0 PASS
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "report",
                str(report_json),
                "--gate",
                "first_pass_success_rate>0.50",
                "--gate",
                "rework_cycles<1.0",
            ],
        )
        assert result.exit_code == 0

    def test_report_gate_shows_quality_gates_heading(self, tmp_path):
        """Gate output should include the 'Quality Gates:' heading."""
        report_json = tmp_path / "report.json"
        _write_report_json(report_json, include_sessions=True)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["report", str(report_json), "--gate", "first_pass_success_rate>0.50"],
        )
        assert result.exit_code == 0
        assert "Quality Gates" in result.output


# ---------------------------------------------------------------------------
# History log tests (ticket #170)
# ---------------------------------------------------------------------------


class TestCliRunHistoryLog:
    """raki run should append one JSONL entry per run to a history file."""

    def test_history_file_created_by_default(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """raki run must create a JSONL history file in .raki/ by default."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        history_file = tmp_path / ".raki" / "history.jsonl"
        assert history_file.exists(), f"Expected {history_file} to be created"

    def test_history_file_contains_one_entry_per_run(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """Each raki run call must append exactly one line to the history file."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        for _ in range(3):
            result = runner.invoke(
                main,
                ["run", "-m", str(manifest), "-o", str(output_dir)],
            )
            assert result.exit_code == 0
        history_file = tmp_path / ".raki" / "history.jsonl"
        lines = history_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_history_entry_contains_metrics(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """Each JSONL entry must contain metrics from the run."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        history_file = tmp_path / ".raki" / "history.jsonl"
        parsed = json.loads(history_file.read_text(encoding="utf-8").strip())
        assert "metrics" in parsed
        assert isinstance(parsed["metrics"], dict)

    def test_history_entry_contains_sessions_count(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """Each JSONL entry must include the sessions_count for that run."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        history_file = tmp_path / ".raki" / "history.jsonl"
        parsed = json.loads(history_file.read_text(encoding="utf-8").strip())
        assert "sessions_count" in parsed
        assert parsed["sessions_count"] >= 0

    def test_history_entry_contains_run_id_and_timestamp(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """Each JSONL entry must have run_id and timestamp fields."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        history_file = tmp_path / ".raki" / "history.jsonl"
        parsed = json.loads(history_file.read_text(encoding="utf-8").strip())
        assert "run_id" in parsed
        assert "timestamp" in parsed

    def test_custom_history_path(self, manifest_with_session, tmp_path, monkeypatch) -> None:
        """--history-path lets the user specify a custom path for the JSONL log."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        custom_history = tmp_path / "custom" / "my-history.jsonl"
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest),
                "-o",
                str(output_dir),
                "--history-path",
                str(custom_history),
            ],
        )
        assert result.exit_code == 0
        assert custom_history.exists(), f"Expected custom history file at {custom_history}"

    def test_no_history_flag_skips_history_file(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """--no-history must suppress creation of the JSONL history file."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir), "--no-history"],
        )
        assert result.exit_code == 0
        history_file = tmp_path / ".raki" / "history.jsonl"
        assert not history_file.exists(), "History file must NOT be created with --no-history"

    def test_no_history_and_history_path_raises_usage_error(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """--no-history + --history-path must raise UsageError."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest),
                "-o",
                str(output_dir),
                "--no-history",
                "--history-path",
                str(tmp_path / "custom.jsonl"),
            ],
        )
        assert result.exit_code != 0
        assert "--no-history" in result.output and "--history-path" in result.output

    def test_history_path_outside_project_root_rejected(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """--history-path pointing outside the CWD (project root) must fail with UsageError."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        manifest, _sessions = manifest_with_session
        outside_history = tmp_path / "outside" / "history.jsonl"
        output_dir = project_dir / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "-m",
                str(manifest),
                "-o",
                str(output_dir),
                "--history-path",
                str(outside_history),
            ],
        )
        assert result.exit_code != 0
        assert "must be within the project root" in result.output

    def test_history_file_reported_in_output(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """raki run should mention the history file path in its output."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        assert "history" in result.output.lower() or "jsonl" in result.output.lower()

    def test_history_file_not_reported_in_quiet_mode(
        self, manifest_with_session, tmp_path, monkeypatch
    ) -> None:
        """In --quiet mode, the history file path must NOT be echoed."""
        monkeypatch.chdir(tmp_path)
        manifest, _sessions = manifest_with_session
        output_dir = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "-m", str(manifest), "-o", str(output_dir), "-q"],
        )
        assert result.exit_code == 0
        # In quiet mode there should be no report-path output at all
        assert "history" not in result.output.lower()


class TestTrendsCommand:
    """Tests for raki trends command."""

    def test_trends_shows_in_help(self) -> None:
        """trends must appear in the main --help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "trends" in result.output

    def test_trends_help_text(self) -> None:
        """trends --help must describe the command."""
        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--help"])
        assert result.exit_code == 0
        assert "history" in result.output.lower()

    def test_trends_no_history_file(self, tmp_path, monkeypatch) -> None:
        """When no history file exists, trends must exit 0 with the exact message."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["trends"])
        assert result.exit_code == 0
        assert "No evaluation history found. Run 'raki run' to generate history." in result.output

    def test_trends_with_history_file(self, tmp_path, monkeypatch) -> None:
        """trends must succeed and display a table when history file exists."""
        from conftest import make_history_entry

        import json

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entry = make_history_entry(metrics={"rework_cycles": 1.5, "first_pass_success_rate": 0.80})
        line = json.dumps(entry.model_dump(mode="json"), default=str)
        history_path.write_text(line + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends"])
        assert result.exit_code == 0
        assert "Trend" in result.output or "Rework" in result.output

    def test_trends_custom_history_path(self, tmp_path, monkeypatch) -> None:
        """--history-path must allow specifying a custom JSONL file."""
        from conftest import make_history_entry

        import json

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / "my-history.jsonl"
        entry = make_history_entry(metrics={"rework_cycles": 1.5})
        line = json.dumps(entry.model_dump(mode="json"), default=str)
        history_path.write_text(line + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--history-path", str(history_path)])
        assert result.exit_code == 0

    def test_trends_json_output(self, tmp_path, monkeypatch) -> None:
        """--json flag must produce valid JSON with a 'trends' key."""
        from conftest import make_history_entry

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entry = make_history_entry(metrics={"rework_cycles": 1.5})
        line = json_mod.dumps(entry.model_dump(mode="json"), default=str)
        history_path.write_text(line + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        assert "trends" in data

    def test_trends_metrics_filter(self, tmp_path, monkeypatch) -> None:
        """--metrics must restrict output to the requested metric(s)."""
        from conftest import make_history_entry

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entry = make_history_entry(metrics={"rework_cycles": 1.5, "first_pass_success_rate": 0.80})
        line = json_mod.dumps(entry.model_dump(mode="json"), default=str)
        history_path.write_text(line + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--metrics", "rework_cycles", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        names = {trend["metric_name"] for trend in data["trends"]}
        assert "rework_cycles" in names
        assert "first_pass_success_rate" not in names

    def test_trends_invalid_metric_name_exits_2(self, tmp_path, monkeypatch) -> None:
        """--metrics with unknown name must exit with code 2."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--metrics", "totally_unknown"])
        assert result.exit_code == 2

    def test_trends_since_filter(self, tmp_path, monkeypatch) -> None:
        """--since must exclude history entries before the given date."""
        from conftest import make_history_entry
        from datetime import datetime, timezone

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        old_entry = make_history_entry(
            run_id="old",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            metrics={"rework_cycles": 3.0},
        )
        new_entry = make_history_entry(
            run_id="new",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            metrics={"rework_cycles": 1.0},
        )
        lines = "\n".join(
            json_mod.dumps(entry.model_dump(mode="json"), default=str)
            for entry in [old_entry, new_entry]
        )
        history_path.write_text(lines + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--since", "2026-03-01", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        rework = next(
            (trend for trend in data["trends"] if trend["metric_name"] == "rework_cycles"), None
        )
        assert rework is not None
        assert rework["run_count"] == 1
        assert rework["values"][0]["value"] == 1.0

    def test_trends_last_n(self, tmp_path, monkeypatch) -> None:
        """--last must limit to the most recent N runs."""
        from conftest import make_history_entry
        from datetime import datetime, timezone

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"rework_cycles": float(idx)},
            )
            for idx in range(5)
        ]
        lines = "\n".join(
            json_mod.dumps(entry.model_dump(mode="json"), default=str) for entry in entries
        )
        history_path.write_text(lines + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--last", "2", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        rework = next(
            (trend for trend in data["trends"] if trend["metric_name"] == "rework_cycles"), None
        )
        assert rework is not None
        assert rework["run_count"] == 2

    def test_trends_last_and_since_conflict(self, tmp_path, monkeypatch) -> None:
        """--last combined with --since must raise UsageError."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--last", "5", "--since", "2026-01-01"])
        assert result.exit_code != 0
        assert "--last" in result.output or "cannot" in result.output.lower()

    def test_trends_last_and_until_conflict(self, tmp_path, monkeypatch) -> None:
        """--last combined with --until must raise UsageError."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--last", "5", "--until", "2026-12-31"])
        assert result.exit_code != 0

    def test_trends_manifest_filter(self, tmp_path, monkeypatch) -> None:
        """--manifest must filter history entries by manifest name."""
        from conftest import make_history_entry
        from datetime import datetime, timezone

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entry_a = make_history_entry(
            run_id="a",
            manifest="raki.yaml",
            metrics={"rework_cycles": 1.5},
        )
        entry_b = make_history_entry(
            run_id="b",
            timestamp=datetime(2026, 4, 2, tzinfo=timezone.utc),
            manifest="other.yaml",
            metrics={"rework_cycles": 2.0},
        )
        lines = "\n".join(
            json_mod.dumps(entry.model_dump(mode="json"), default=str)
            for entry in [entry_a, entry_b]
        )
        history_path.write_text(lines + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--manifest", "raki.yaml", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        rework = next(
            (trend for trend in data["trends"] if trend["metric_name"] == "rework_cycles"),
            None,
        )
        assert rework is not None
        assert rework["run_count"] == 1
        assert rework["values"][0]["value"] == 1.5

    def test_trends_manifest_filter_no_matches(self, tmp_path, monkeypatch) -> None:
        """--manifest with no matching entries must show empty history message."""
        from conftest import make_history_entry

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entry = make_history_entry(manifest="other.yaml", metrics={"rework_cycles": 1.5})
        line = json_mod.dumps(entry.model_dump(mode="json"), default=str)
        history_path.write_text(line + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--manifest", "raki.yaml", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        assert data["trends"] == []

    def test_trends_default_last_20(self, tmp_path, monkeypatch) -> None:
        """Default --last should limit to 20 most recent entries."""
        from conftest import make_history_entry
        from datetime import datetime, timezone

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 1, 1 + idx, tzinfo=timezone.utc),
                metrics={"rework_cycles": float(idx)},
            )
            for idx in range(25)
        ]
        lines = "\n".join(
            json_mod.dumps(entry.model_dump(mode="json"), default=str) for entry in entries
        )
        history_path.write_text(lines + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        rework = next(
            (trend for trend in data["trends"] if trend["metric_name"] == "rework_cycles"),
            None,
        )
        assert rework is not None
        # Default --last=20 limits to 20 entries
        assert rework["run_count"] == 20

    def test_trends_since_ignores_default_last(self, tmp_path, monkeypatch) -> None:
        """When --since is used without explicit --last, default --last is disabled."""
        from conftest import make_history_entry
        from datetime import datetime, timezone

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, 1 + idx, tzinfo=timezone.utc),
                metrics={"rework_cycles": float(idx)},
            )
            for idx in range(25)
        ]
        lines = "\n".join(
            json_mod.dumps(entry.model_dump(mode="json"), default=str) for entry in entries
        )
        history_path.write_text(lines + "\n")

        runner = CliRunner()
        # --since without --last: should include all entries after the date
        result = runner.invoke(main, ["trends", "--since", "2026-04-01", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        rework = next(
            (trend for trend in data["trends"] if trend["metric_name"] == "rework_cycles"),
            None,
        )
        assert rework is not None
        assert rework["run_count"] == 25

    def test_trends_direction_in_json(self, tmp_path, monkeypatch) -> None:
        """JSON output must include the 'direction' field for each trend."""
        from conftest import make_history_entry
        from datetime import datetime, timezone

        import json as json_mod

        monkeypatch.chdir(tmp_path)
        history_path = tmp_path / ".raki" / "history.jsonl"
        history_path.parent.mkdir(parents=True)

        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"first_pass_success_rate": 0.60 + idx * 0.10},
            )
            for idx in range(3)
        ]
        lines = "\n".join(
            json_mod.dumps(entry.model_dump(mode="json"), default=str) for entry in entries
        )
        history_path.write_text(lines + "\n")

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        fps = next(
            (
                trend
                for trend in data["trends"]
                if trend["metric_name"] == "first_pass_success_rate"
            ),
            None,
        )
        assert fps is not None
        assert fps["direction"] == "improving"
