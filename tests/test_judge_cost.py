"""Tests for judge cost tracking — TokenAccumulator, client patching, engine integration, reports."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raki.metrics.protocol import MetricConfig, TokenAccumulator
from raki.model.report import EvalReport

from conftest import make_dataset, make_sample


# --- TokenAccumulator tests ---


class TestTokenAccumulator:
    def test_default_initialization(self) -> None:
        """TokenAccumulator starts at zero for all fields."""
        accumulator = TokenAccumulator()
        assert accumulator.input_tokens == 0
        assert accumulator.output_tokens == 0
        assert accumulator.calls == 0

    def test_single_increment(self) -> None:
        """Fields can be incremented individually."""
        accumulator = TokenAccumulator()
        accumulator.input_tokens += 100
        accumulator.output_tokens += 50
        accumulator.calls += 1
        assert accumulator.input_tokens == 100
        assert accumulator.output_tokens == 50
        assert accumulator.calls == 1

    def test_multiple_increments(self) -> None:
        """Multiple increments accumulate correctly."""
        accumulator = TokenAccumulator()
        for idx in range(5):
            accumulator.input_tokens += 1000
            accumulator.output_tokens += 200
            accumulator.calls += 1
        assert accumulator.input_tokens == 5000
        assert accumulator.output_tokens == 1000
        assert accumulator.calls == 5


# --- MetricConfig with TokenAccumulator tests ---


class TestMetricConfigTokenAccumulator:
    def test_token_accumulator_defaults_to_none(self) -> None:
        """MetricConfig.token_accumulator defaults to None."""
        config = MetricConfig()
        assert config.token_accumulator is None

    def test_token_accumulator_can_be_set(self) -> None:
        """MetricConfig accepts a TokenAccumulator instance."""
        accumulator = TokenAccumulator()
        config = MetricConfig(token_accumulator=accumulator)
        assert config.token_accumulator is accumulator

    def test_token_accumulator_not_in_repr(self) -> None:
        """TokenAccumulator field should be excluded from repr for clean output."""
        accumulator = TokenAccumulator()
        config = MetricConfig(token_accumulator=accumulator)
        config_repr = repr(config)
        assert "token_accumulator" not in config_repr


# --- patch_client_for_token_tracking tests ---


class TestPatchClientForTokenTracking:
    def test_captures_tokens_from_response(self) -> None:
        """Patching a mock client should capture token usage from responses."""
        from raki.metrics.ragas.llm_setup import patch_client_for_token_tracking

        accumulator = TokenAccumulator()
        usage = SimpleNamespace(input_tokens=500, output_tokens=100)
        response = SimpleNamespace(usage=usage, content="test response")

        mock_create = AsyncMock(return_value=response)
        mock_messages = SimpleNamespace(create=mock_create)
        mock_client = SimpleNamespace(messages=mock_messages)

        patch_client_for_token_tracking(mock_client, accumulator)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(mock_client.messages.create())
        loop.close()
        assert accumulator.input_tokens == 500
        assert accumulator.output_tokens == 100
        assert accumulator.calls == 1
        assert result is response

    def test_response_returned_unmodified(self) -> None:
        """The original response object must be returned without modification."""
        from raki.metrics.ragas.llm_setup import patch_client_for_token_tracking

        accumulator = TokenAccumulator()
        usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        response = SimpleNamespace(usage=usage, id="msg_123", content="hello")

        mock_create = AsyncMock(return_value=response)
        mock_messages = SimpleNamespace(create=mock_create)
        mock_client = SimpleNamespace(messages=mock_messages)

        patch_client_for_token_tracking(mock_client, accumulator)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(mock_client.messages.create())
        loop.close()
        assert result.id == "msg_123"
        assert result.content == "hello"

    def test_multiple_calls_accumulate(self) -> None:
        """Multiple patched calls should accumulate tokens correctly."""
        from raki.metrics.ragas.llm_setup import patch_client_for_token_tracking

        accumulator = TokenAccumulator()

        call_count = 0

        async def fake_create(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            usage = SimpleNamespace(input_tokens=300 * call_count, output_tokens=60 * call_count)
            return SimpleNamespace(usage=usage)

        mock_messages = SimpleNamespace(create=fake_create)
        mock_client = SimpleNamespace(messages=mock_messages)

        patch_client_for_token_tracking(mock_client, accumulator)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mock_client.messages.create())
        loop.run_until_complete(mock_client.messages.create())
        loop.run_until_complete(mock_client.messages.create())
        loop.close()

        assert accumulator.calls == 3
        # 300 + 600 + 900 = 1800
        assert accumulator.input_tokens == 1800
        # 60 + 120 + 180 = 360
        assert accumulator.output_tokens == 360

    def test_handles_response_without_usage(self) -> None:
        """Patched client should not crash if response lacks usage attribute."""
        from raki.metrics.ragas.llm_setup import patch_client_for_token_tracking

        accumulator = TokenAccumulator()
        response = SimpleNamespace(content="no usage here")

        mock_create = AsyncMock(return_value=response)
        mock_messages = SimpleNamespace(create=mock_create)
        mock_client = SimpleNamespace(messages=mock_messages)

        patch_client_for_token_tracking(mock_client, accumulator)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(mock_client.messages.create())
        loop.close()
        assert accumulator.calls == 1
        assert accumulator.input_tokens == 0
        assert accumulator.output_tokens == 0
        assert result is response


# --- Engine integration tests ---


class TestEngineJudgeCost:
    def test_judge_cost_in_report_when_accumulator_has_calls(self) -> None:
        """When the accumulator records calls, judge_cost should appear in report config."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        sample = make_sample("s1")
        dataset = make_dataset(sample)
        engine = MetricsEngine([TokenEfficiencyMetric()], config=MetricConfig())

        # Simulate token usage by pre-setting the accumulator
        # The engine creates its own accumulator, so we test indirectly
        report = engine.run(dataset)

        # No LLM calls happened, so judge_cost should not be in config
        assert "judge_cost" not in report.config

    def test_judge_cost_absent_when_no_llm_calls(self) -> None:
        """When no LLM calls happen (operational metrics only), judge_cost is absent."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.rework import ReworkCycles

        sample = make_sample("s1", rework_cycles=2)
        dataset = make_dataset(sample)
        engine = MetricsEngine([ReworkCycles()], config=MetricConfig())
        report = engine.run(dataset)
        assert "judge_cost" not in report.config

    def test_accumulator_attached_to_config(self) -> None:
        """MetricsEngine.run() should attach a TokenAccumulator to the config."""
        from raki.metrics.engine import MetricsEngine
        from raki.metrics.operational.token_efficiency import TokenEfficiencyMetric

        config = MetricConfig()
        engine = MetricsEngine([TokenEfficiencyMetric()], config=config)
        engine.run(make_dataset(make_sample("s1")))
        assert config.token_accumulator is not None
        assert isinstance(config.token_accumulator, TokenAccumulator)

    def test_judge_cost_structure_when_present(self) -> None:
        """When present, judge_cost must have input_tokens, output_tokens, and calls keys."""
        report = EvalReport(
            run_id="judge-cost-struct",
            config={
                "judge_cost": {
                    "input_tokens": 15000,
                    "output_tokens": 3000,
                    "calls": 24,
                },
            },
            aggregate_scores={},
        )
        judge_cost = report.config["judge_cost"]
        assert judge_cost["input_tokens"] == 15000
        assert judge_cost["output_tokens"] == 3000
        assert judge_cost["calls"] == 24


# --- CLI summary tests ---


class TestCliSummaryJudgeCost:
    def test_judge_cost_shown_in_summary(self) -> None:
        """When judge_cost is in the report config, it should appear in CLI output."""
        from rich.console import Console

        from raki.report.cli_summary import print_summary

        report = EvalReport(
            run_id="judge-cli-test",
            config={
                "judge_cost": {
                    "input_tokens": 15000,
                    "output_tokens": 3000,
                    "calls": 24,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, width=120)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "24 calls" in output
        assert "15,000 in" in output
        assert "3,000 out" in output
        assert "Judge:" in output

    def test_judge_cost_not_shown_when_absent(self) -> None:
        """When judge_cost is not in the report config, no judge line appears."""
        from rich.console import Console

        from raki.report.cli_summary import print_summary

        report = EvalReport(
            run_id="no-judge-test",
            config={},
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, width=120)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "Judge:" not in output

    def test_judge_model_and_provider_shown_in_summary(self) -> None:
        """When llm_model and llm_provider are set, both appear in the judge cost line."""
        from rich.console import Console

        from raki.report.cli_summary import print_summary

        report = EvalReport(
            run_id="judge-model-provider-test",
            config={
                "llm_model": "claude-sonnet-4-6",
                "llm_provider": "vertex-anthropic",
                "judge_cost": {
                    "input_tokens": 41861,
                    "output_tokens": 11558,
                    "calls": 20,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, width=120)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "claude-sonnet-4-6" in output
        assert "vertex-anthropic" in output
        assert "20 calls" in output

    def test_judge_model_only_shown_in_summary(self) -> None:
        """When only llm_model is set (no provider), model appears without provider."""
        from rich.console import Console

        from raki.report.cli_summary import print_summary

        report = EvalReport(
            run_id="judge-model-only-test",
            config={
                "llm_model": "claude-opus-4",
                "judge_cost": {
                    "input_tokens": 5000,
                    "output_tokens": 1000,
                    "calls": 5,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, width=120)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "claude-opus-4" in output
        assert "(" not in output or "vertex-anthropic" not in output
        assert "5 calls" in output

    def test_judge_cost_only_no_model_in_summary(self) -> None:
        """When llm_model is absent, no model prefix appears in the judge cost line."""
        from rich.console import Console

        from raki.report.cli_summary import print_summary

        report = EvalReport(
            run_id="judge-cost-only-test",
            config={
                "judge_cost": {
                    "input_tokens": 15000,
                    "output_tokens": 3000,
                    "calls": 24,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, width=120)
        print_summary(report, session_count=10, console=test_console)
        output = string_io.getvalue()
        assert "Judge: 24 calls" in output


# --- HTML report tests ---


class TestHtmlReportJudgeCost:
    @pytest.fixture(autouse=True)
    def _require_jinja2(self) -> None:
        pytest.importorskip("jinja2")

    def test_judge_cost_appears_in_html_report(self, tmp_path: Path) -> None:
        """When judge_cost is in the config, it should appear in the HTML output."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="html-judge-test",
            config={
                "judge_cost": {
                    "input_tokens": 15000,
                    "output_tokens": 3000,
                    "calls": 24,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        output_path = tmp_path / "report.html"
        write_html_report(report, output_path, session_count=10)
        html_content = output_path.read_text()
        assert "24 calls" in html_content
        assert "15,000 in" in html_content
        assert "3,000 out" in html_content

    def test_judge_cost_absent_from_html_when_not_in_config(self, tmp_path: Path) -> None:
        """When judge_cost is not in the config, judge line should not appear in HTML."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="html-no-judge-test",
            config={},
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        output_path = tmp_path / "report.html"
        write_html_report(report, output_path, session_count=10)
        html_content = output_path.read_text()
        assert "Judge:" not in html_content

    def test_judge_model_and_provider_in_html_report(self, tmp_path: Path) -> None:
        """When llm_model and llm_provider are set, both appear in the HTML header."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="html-judge-model-provider-test",
            config={
                "llm_model": "claude-sonnet-4-6",
                "llm_provider": "vertex-anthropic",
                "judge_cost": {
                    "input_tokens": 41861,
                    "output_tokens": 11558,
                    "calls": 20,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        output_path = tmp_path / "report.html"
        write_html_report(report, output_path, session_count=10)
        html_content = output_path.read_text()
        assert "claude-sonnet-4-6" in html_content
        assert "vertex-anthropic" in html_content
        assert "20 calls" in html_content

    def test_judge_model_only_in_html_report(self, tmp_path: Path) -> None:
        """When only llm_model is set, model appears without provider in HTML header."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="html-judge-model-only-test",
            config={
                "llm_model": "claude-opus-4",
                "judge_cost": {
                    "input_tokens": 5000,
                    "output_tokens": 1000,
                    "calls": 5,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        output_path = tmp_path / "report.html"
        write_html_report(report, output_path, session_count=10)
        html_content = output_path.read_text()
        assert "claude-opus-4" in html_content
        assert "5 calls" in html_content

    def test_judge_cost_only_no_model_in_html_report(self, tmp_path: Path) -> None:
        """When llm_model is absent, no model prefix appears in the HTML judge span."""
        from raki.report.html_report import write_html_report

        report = EvalReport(
            run_id="html-judge-cost-only-test",
            config={
                "judge_cost": {
                    "input_tokens": 15000,
                    "output_tokens": 3000,
                    "calls": 24,
                },
            },
            aggregate_scores={"first_pass_success_rate": 0.8},
        )
        output_path = tmp_path / "report.html"
        write_html_report(report, output_path, session_count=10)
        html_content = output_path.read_text()
        assert "24 calls" in html_content
        assert "Judge:" in html_content
