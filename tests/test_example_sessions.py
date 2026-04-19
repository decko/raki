"""Tests that verify example session data loads correctly through adapters."""

from pathlib import Path

import pytest

from raki.adapters.alcove import AlcoveAdapter
from raki.adapters.session_schema import SessionSchemaAdapter
from raki.model import EvalSample

EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "sessions"

SESSION_SCHEMA_SESSIONS = [
    "pass-clean",
    "rework-cycle",
    "critical-findings",
    "partial-failure",
    "multi-reviewer",
]


@pytest.fixture
def session_adapter() -> SessionSchemaAdapter:
    return SessionSchemaAdapter()


@pytest.fixture
def alcove_adapter() -> AlcoveAdapter:
    return AlcoveAdapter()


@pytest.mark.parametrize("session_name", SESSION_SCHEMA_SESSIONS)
def test_session_schema_example_detects(
    session_adapter: SessionSchemaAdapter, session_name: str
) -> None:
    """Each session-schema example directory must be detected by the adapter."""
    session_path = EXAMPLES_DIR / session_name
    assert session_adapter.detect(session_path), f"{session_name} not detected"


@pytest.mark.parametrize("session_name", SESSION_SCHEMA_SESSIONS)
def test_session_schema_example_loads_as_eval_sample(
    session_adapter: SessionSchemaAdapter, session_name: str
) -> None:
    """Each session-schema example must load into a valid EvalSample."""
    session_path = EXAMPLES_DIR / session_name
    sample = session_adapter.load(session_path)
    assert isinstance(sample, EvalSample)
    assert sample.session.session_id is not None
    assert len(sample.events) > 0


@pytest.mark.parametrize("session_name", SESSION_SCHEMA_SESSIONS)
def test_session_schema_example_has_readme(session_name: str) -> None:
    """Each session-schema example directory must contain a _README.md."""
    readme_path = EXAMPLES_DIR / session_name / "_README.md"
    assert readme_path.exists(), f"{session_name} missing _README.md"
    content = readme_path.read_text()
    lines = [line for line in content.strip().splitlines() if line.strip()]
    assert 3 <= len(lines) <= 6, f"{session_name} _README.md should have 3-5 content lines"


def test_pass_clean_has_no_rework(session_adapter: SessionSchemaAdapter) -> None:
    """pass-clean session should have zero rework cycles."""
    sample = session_adapter.load(EXAMPLES_DIR / "pass-clean")
    assert sample.session.rework_cycles == 0


def test_rework_cycle_has_generational_files(session_adapter: SessionSchemaAdapter) -> None:
    """rework-cycle should have multiple implement generations."""
    sample = session_adapter.load(EXAMPLES_DIR / "rework-cycle")
    implement_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(implement_phases) == 2, "Expected 2 implement generations"
    generations = sorted(phase.generation for phase in implement_phases)
    assert generations == [1, 2]


def test_rework_cycle_events_cover_full_taxonomy(session_adapter: SessionSchemaAdapter) -> None:
    """rework-cycle events.jsonl must cover the full event taxonomy."""
    sample = session_adapter.load(EXAMPLES_DIR / "rework-cycle")
    event_kinds = {event.kind for event in sample.events}
    required_kinds = {
        "phase_started",
        "phase_completed",
        "phase_failed",
        "rework_feedback_injected",
        "reviewer_started",
        "reviewer_completed",
        "review_merged",
    }
    missing = required_kinds - event_kinds
    assert not missing, f"rework-cycle events missing kinds: {missing}"


def test_rework_cycle_has_rework_cycles(session_adapter: SessionSchemaAdapter) -> None:
    """rework-cycle session should have at least 1 rework cycle."""
    sample = session_adapter.load(EXAMPLES_DIR / "rework-cycle")
    assert sample.session.rework_cycles >= 1


def test_critical_findings_has_critical_severity(session_adapter: SessionSchemaAdapter) -> None:
    """critical-findings session should have findings with severity:critical."""
    sample = session_adapter.load(EXAMPLES_DIR / "critical-findings")
    critical_findings = [finding for finding in sample.findings if finding.severity == "critical"]
    assert len(critical_findings) >= 1, "Expected at least one critical finding"


def test_critical_findings_has_multiple_reviewers(session_adapter: SessionSchemaAdapter) -> None:
    """critical-findings session should have findings from multiple reviewers."""
    sample = session_adapter.load(EXAMPLES_DIR / "critical-findings")
    reviewers = {finding.reviewer for finding in sample.findings}
    assert len(reviewers) >= 2, f"Expected multiple reviewers, got: {reviewers}"


def test_partial_failure_has_failed_phase(session_adapter: SessionSchemaAdapter) -> None:
    """partial-failure session should have a failed implement phase."""
    sample = session_adapter.load(EXAMPLES_DIR / "partial-failure")
    failed_events = [event for event in sample.events if event.kind == "phase_failed"]
    assert len(failed_events) >= 1, "Expected at least one phase_failed event"


def test_partial_failure_has_no_verify(session_adapter: SessionSchemaAdapter) -> None:
    """partial-failure should have no verify phase files."""
    sample = session_adapter.load(EXAMPLES_DIR / "partial-failure")
    verify_phases = [phase for phase in sample.phases if phase.name == "verify"]
    assert len(verify_phases) == 0, "partial-failure should not have verify phases"


def test_multi_reviewer_has_review_merged(session_adapter: SessionSchemaAdapter) -> None:
    """multi-reviewer session should have a review_merged event."""
    sample = session_adapter.load(EXAMPLES_DIR / "multi-reviewer")
    merged_events = [event for event in sample.events if event.kind == "review_merged"]
    assert len(merged_events) >= 1, "Expected at least one review_merged event"


def test_multi_reviewer_has_parallel_reviewers(session_adapter: SessionSchemaAdapter) -> None:
    """multi-reviewer session should have reviewer_started events for two reviewers."""
    sample = session_adapter.load(EXAMPLES_DIR / "multi-reviewer")
    reviewer_started = [event for event in sample.events if event.kind == "reviewer_started"]
    reviewer_names = {event.data.get("reviewer") for event in reviewer_started}
    assert len(reviewer_names) >= 2, f"Expected 2+ reviewers, got: {reviewer_names}"


def test_alcove_session_detects(alcove_adapter: AlcoveAdapter) -> None:
    """Alcove example file must be detected by the adapter."""
    alcove_path = EXAMPLES_DIR / "alcove-session.json"
    assert alcove_adapter.detect(alcove_path), "alcove-session.json not detected"


def test_alcove_session_loads_as_eval_sample(alcove_adapter: AlcoveAdapter) -> None:
    """Alcove example must load into a valid EvalSample."""
    sample = alcove_adapter.load(EXAMPLES_DIR / "alcove-session.json")
    assert isinstance(sample, EvalSample)
    assert sample.session.session_id == "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def test_alcove_session_has_tool_calls(alcove_adapter: AlcoveAdapter) -> None:
    """Alcove example should have tool calls extracted."""
    sample = alcove_adapter.load(EXAMPLES_DIR / "alcove-session.json")
    assert len(sample.phases) == 1
    tool_calls = sample.phases[0].tool_calls
    assert len(tool_calls) >= 1, "Expected at least one tool call"
    tool_names = {call.name for call in tool_calls}
    assert "Bash" in tool_names or "Read" in tool_names or "Edit" in tool_names


def test_alcove_session_has_token_usage(alcove_adapter: AlcoveAdapter) -> None:
    """Alcove example should have token usage data."""
    sample = alcove_adapter.load(EXAMPLES_DIR / "alcove-session.json")
    phase = sample.phases[0]
    assert phase.tokens_in is not None and phase.tokens_in > 0
    assert phase.tokens_out is not None and phase.tokens_out > 0


def test_alcove_session_has_cost(alcove_adapter: AlcoveAdapter) -> None:
    """Alcove example should have cost information."""
    sample = alcove_adapter.load(EXAMPLES_DIR / "alcove-session.json")
    assert sample.session.total_cost_usd is not None
    assert sample.session.total_cost_usd > 0
