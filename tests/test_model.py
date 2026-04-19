from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from raki.model import (
    EvalDataset,
    EvalReport,
    EvalSample,
    GroundTruth,
    MetricResult,
    PhaseResult,
    ReviewFinding,
    SessionEvent,
    SessionMeta,
    ToolCall,
)


def test_session_meta_required_fields():
    meta = SessionMeta(
        session_id="101",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=5,
        rework_cycles=0,
    )
    assert meta.session_id == "101"
    assert meta.tenant_id is None
    assert meta.ticket is None
    assert meta.knowledge_version is None
    assert meta.model_id is None
    assert meta.total_cost_usd is None


def test_session_meta_all_fields():
    meta = SessionMeta(
        session_id="53",
        tenant_id="team-pulp",
        ticket="53",
        started_at=datetime(2026, 4, 16, 8, 29, tzinfo=timezone.utc),
        total_cost_usd=26.1,
        total_phases=5,
        rework_cycles=2,
        knowledge_version="abc123",
        model_id="claude-opus-4-6",
    )
    assert meta.total_cost_usd == 26.1
    assert meta.knowledge_version == "abc123"


def test_phase_result_minimal():
    phase = PhaseResult(
        name="triage",
        generation=1,
        status="completed",
        output="small complexity",
    )
    assert phase.cost_usd is None
    assert phase.tool_calls == []
    assert phase.files_modified == []
    assert phase.knowledge_context is None
    assert phase.instruction_context is None


def test_phase_result_with_tool_calls():
    tool_call = ToolCall(name="bash", arguments={"command": "go test ./..."})
    phase = PhaseResult(
        name="verify",
        generation=1,
        status="completed",
        output="PASS",
        tool_calls=[tool_call],
        files_modified=["engine.go", "engine_test.go"],
    )
    assert len(phase.tool_calls) == 1
    assert phase.tool_calls[0].name == "bash"
    assert len(phase.files_modified) == 2


def test_review_finding():
    finding = ReviewFinding(
        reviewer="go-specialist",
        severity="critical",
        file="cmd/soda/run.go",
        line=294,
        issue="Send on closed channel panic",
        suggestion="Remove defer close(pauseSignal)",
    )
    assert finding.severity == "critical"


def test_session_event():
    event = SessionEvent(
        timestamp=datetime(2026, 4, 16, 8, 29, tzinfo=timezone.utc),
        phase="triage",
        kind="phase_started",
        data={"generation": 1},
    )
    assert event.kind == "phase_started"


def test_eval_sample_assembly():
    meta = SessionMeta(
        session_id="101",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=2,
        rework_cycles=0,
    )
    phase = PhaseResult(name="triage", generation=1, status="completed", output="small")
    finding = ReviewFinding(reviewer="go-specialist", severity="minor", issue="nit")
    sample = EvalSample(session=meta, phases=[phase], findings=[finding], events=[])
    assert sample.ground_truth is None
    assert len(sample.phases) == 1
    assert len(sample.findings) == 1


def test_eval_dataset():
    meta = SessionMeta(
        session_id="101",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )
    sample = EvalSample(session=meta, phases=[], findings=[], events=[])
    dataset = EvalDataset(samples=[sample])
    assert len(dataset.samples) == 1
    assert dataset.manifest_hash is None


def test_eval_report_serialization():
    report = EvalReport(
        run_id="eval-2026-04-17-abc",
        config={"adapter": "session-schema"},
        aggregate_scores={"rework_cycles": 0.7},
        sample_results=[],
    )
    dumped = report.model_dump()
    restored = EvalReport.model_validate(dumped)
    assert restored.run_id == report.run_id
    assert restored.aggregate_scores == report.aggregate_scores


def test_metric_result_with_sample_scores():
    result = MetricResult(
        name="first_pass_verify_rate",
        score=0.58,
        details={"passed": 22, "total": 38},
        sample_scores={"101": 1.0, "53": 0.0},
    )
    assert result.sample_scores["101"] == 1.0


# --- Negative-path tests for Literal constraints ---


def test_review_finding_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        ReviewFinding(
            reviewer="go-specialist",
            severity="bogus",
            issue="bad severity value",
        )


def test_ground_truth_rejects_invalid_difficulty():
    with pytest.raises(ValidationError):
        GroundTruth(difficulty="impossible")


def test_ground_truth_rejects_invalid_knowledge_type():
    with pytest.raises(ValidationError):
        GroundTruth(knowledge_type="magic")


def test_phase_result_rejects_invalid_status():
    with pytest.raises(ValidationError):
        PhaseResult(
            name="triage",
            generation=1,
            status="bogus_status",
            output="should fail",
        )


def test_session_event_rejects_invalid_kind():
    with pytest.raises(ValidationError):
        SessionEvent(
            timestamp=datetime(2026, 4, 16, 8, 29, tzinfo=timezone.utc),
            kind="invalid_event_kind",
        )


# --- Standalone GroundTruth test ---


def test_ground_truth():
    """Verify GroundTruth fields, defaults, domain isolation, and Literal constraints."""
    ground_truth = GroundTruth(
        question="How does the adapter detect format?",
        expected_approach="Read first 4KB header",
        expected_files=["src/raki/adapters/detect.py"],
        expected_contexts=["adapter-detection"],
        acceptance_criteria=["Must detect JSON and YAML"],
        reference_answer="Sniff the first 4KB and match signatures.",
        domains=["adapters", "detection"],
        difficulty="hard",
        knowledge_type="procedure",
        expected_phase="triage",
    )
    assert ground_truth.question == "How does the adapter detect format?"
    assert ground_truth.expected_approach == "Read first 4KB header"
    assert ground_truth.expected_files == ["src/raki/adapters/detect.py"]
    assert ground_truth.expected_contexts == ["adapter-detection"]
    assert ground_truth.acceptance_criteria == ["Must detect JSON and YAML"]
    assert ground_truth.reference_answer == "Sniff the first 4KB and match signatures."
    assert ground_truth.domains == ["adapters", "detection"]
    assert ground_truth.difficulty == "hard"
    assert ground_truth.knowledge_type == "procedure"
    assert ground_truth.expected_phase == "triage"

    # Defaults: all optional fields are None, domains defaults to empty list
    minimal = GroundTruth()
    assert minimal.question is None
    assert minimal.expected_approach is None
    assert minimal.expected_files is None
    assert minimal.expected_contexts is None
    assert minimal.acceptance_criteria is None
    assert minimal.reference_answer is None
    assert minimal.domains == []
    assert minimal.difficulty is None
    assert minimal.knowledge_type is None
    assert minimal.expected_phase is None

    # default_factory isolation: mutating one instance's domains must not affect another
    instance_a = GroundTruth()
    instance_b = GroundTruth()
    instance_a.domains.append("leaked")
    assert instance_b.domains == []

    # Literal constraints validated above in dedicated tests;
    # verify valid Literal values are accepted
    for valid_difficulty in ("easy", "medium", "hard"):
        truth = GroundTruth(difficulty=valid_difficulty)
        assert truth.difficulty == valid_difficulty

    for valid_knowledge in ("fact", "procedure", "constraint", "context-dependent"):
        truth = GroundTruth(knowledge_type=valid_knowledge)
        assert truth.knowledge_type == valid_knowledge
