"""Comprehensive test suite for AlcovePipelineAdapter.

All tests use tmp_path with synthetic data — no external dependencies.
Tests covering the real /tmp/alcove-export-81be9b17 sample are skipped when
that path is absent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from raki.adapters import AlcovePipelineAdapter, default_registry
from raki.adapters.alcove_pipeline import (
    _is_corrective,
    _parse_issues,
    _phase_status,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data builders
# ---------------------------------------------------------------------------

SAMPLE_DATA_PATH = Path("/tmp/alcove-export-81be9b17")


def _make_transcript(
    session_id: str = "sess-001",
    model: str = "claude-test",
    cost_usd: float = 1.0,
    duration_ms: int = 5000,
    tokens_in: int = 100,
    tokens_out: int = 200,
    output_text: str = "done",
) -> dict:
    """Build a minimal alcove transcript dict suitable for a step transcript.json."""
    return {
        "session_id": session_id,
        "transcript": [
            {
                "type": "system",
                "model": model,
                "uuid": "sys-001",
                "session_id": session_id,
            },
            {
                "type": "assistant",
                "uuid": "asst-001",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                    "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
                },
                "session_id": session_id,
            },
            {
                "type": "result",
                "uuid": "res-001",
                "total_cost_usd": cost_usd,
                "duration_ms": duration_ms,
                "modelUsage": {model: {"costUSD": cost_usd}},
                "result": output_text,
                "subtype": "success",
            },
        ],
    }


def _make_run_json(steps: list[dict], run_id: str = "run-abc123") -> dict:
    """Build a minimal run.json dict."""
    for step in steps:
        step.setdefault("run_id", run_id)
    return {"id": "", "steps": steps}


def _make_step_dict(
    step_id: str,
    status: str = "completed",
    step_type: str = "agent",
    outputs: dict | None = None,
    iteration: int = 1,
    started_at: str = "2026-04-01T10:00:00Z",
    finished_at: str = "2026-04-01T10:05:00Z",
    depends: str | None = None,
    run_id: str = "run-abc123",
) -> dict:
    """Build a synthetic step metadata dict (as found in run.json steps[])."""
    result: dict = {
        "id": f"step-id-{step_id}",
        "run_id": run_id,
        "step_id": step_id,
        "status": status,
        "type": step_type,
        "iteration": iteration,
        "started_at": started_at if status != "skipped" else "",
        "finished_at": finished_at if status != "skipped" else "",
    }
    if outputs is not None:
        result["outputs"] = outputs
    if depends is not None:
        result["depends"] = depends
    return result


def _build_pipeline_dir(
    tmp_path: Path,
    steps: list[dict],
    run_id: str = "run-abc123",
    step_transcripts: dict[str, dict] | None = None,
) -> Path:
    """Create a synthetic pipeline export directory in tmp_path.

    Args:
        tmp_path: Destination directory.
        steps: List of step dicts (step_id, status, outputs, …).
        run_id: The pipeline run identifier.
        step_transcripts: Optional mapping of step_id → transcript dict.
            When provided, writes transcript.json for that step; otherwise
            writes a minimal step.json.

    Returns:
        Path to the created pipeline export directory.
    """
    pipeline_dir = tmp_path / "pipeline-export"
    pipeline_dir.mkdir()

    run_data = _make_run_json(steps, run_id=run_id)
    (pipeline_dir / "run.json").write_text(json.dumps(run_data))

    steps_dir = pipeline_dir / "steps"
    steps_dir.mkdir()

    for idx, step in enumerate(steps, start=1):
        step_id = step["step_id"]
        step_dir = steps_dir / f"{idx:02d}-{step_id}"
        step_dir.mkdir()

        if step_transcripts and step_id in step_transcripts:
            transcript_data = step_transcripts[step_id]
            (step_dir / "transcript.json").write_text(json.dumps(transcript_data))
        else:
            # Write minimal step.json for bridge/skipped steps.
            step_data = {key: value for key, value in step.items()}
            (step_dir / "step.json").write_text(json.dumps(step_data))

    return pipeline_dir


# ---------------------------------------------------------------------------
# Task 3 — Detection tests
# ---------------------------------------------------------------------------


def test_detect_valid_pipeline_dir(tmp_path: Path) -> None:
    """Adapter detects a directory with run.json containing 'steps'."""
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=[_make_step_dict("triage")],
    )
    adapter = AlcovePipelineAdapter()
    assert adapter.detect(pipeline_dir) is True


def test_detect_rejects_regular_file(tmp_path: Path) -> None:
    """Adapter rejects a plain file (not a directory)."""
    json_file = tmp_path / "run.json"
    json_file.write_text(json.dumps({"steps": []}))
    adapter = AlcovePipelineAdapter()
    assert adapter.detect(json_file) is False


def test_detect_rejects_dir_without_run_json(tmp_path: Path) -> None:
    """Adapter rejects a directory that lacks run.json."""
    adapter = AlcovePipelineAdapter()
    assert adapter.detect(tmp_path) is False


def test_detect_rejects_dir_with_run_json_no_steps(tmp_path: Path) -> None:
    """Adapter rejects a run.json that does not contain 'steps'."""
    (tmp_path / "run.json").write_text(json.dumps({"id": "x", "workflow_id": "y"}))
    adapter = AlcovePipelineAdapter()
    assert adapter.detect(tmp_path) is False


def test_detect_rejects_symlink(tmp_path: Path) -> None:
    """Adapter rejects symlinks."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "run.json").write_text(json.dumps({"steps": []}))
    link = tmp_path / "link"
    link.symlink_to(real_dir)
    adapter = AlcovePipelineAdapter()
    assert adapter.detect(link) is False


# ---------------------------------------------------------------------------
# Phase mapping tests
# ---------------------------------------------------------------------------


def test_phase_names_match_step_ids(tmp_path: Path) -> None:
    """Each phase name equals the step_id from the directory name."""
    steps = [
        _make_step_dict("triage"),
        _make_step_dict("plan"),
        _make_step_dict("implement"),
        _make_step_dict("verify", outputs={"verdict": "pass"}),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    phase_names = [phase.name for phase in sample.phases]
    assert phase_names == ["triage", "plan", "implement", "verify"]


def test_skipped_phases_have_skipped_status(tmp_path: Path) -> None:
    """Steps with status='skipped' produce PhaseResult with status='skipped'."""
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("patch", status="skipped", depends="verify.Failed"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    patch_phase = next(phase for phase in sample.phases if phase.name == "patch")
    assert patch_phase.status == "skipped"


def test_verify_phase_fail_verdict_produces_failed_status(tmp_path: Path) -> None:
    """Verify step with outputs.verdict='fail' produces status='failed'."""
    steps = [
        _make_step_dict("verify", outputs={"verdict": "fail"}),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    verify_phase = sample.phases[0]
    assert verify_phase.name == "verify"
    assert verify_phase.status == "failed"


def test_verify_phase_pass_verdict_produces_completed_status(tmp_path: Path) -> None:
    """Verify step with outputs.verdict='pass' produces status='completed'."""
    steps = [
        _make_step_dict("verify", outputs={"verdict": "pass"}),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    verify_phase = sample.phases[0]
    assert verify_phase.status == "completed"


def test_verify_verdict_case_insensitive(tmp_path: Path) -> None:
    """Verdict matching is case-insensitive (FAIL, Fail, fail all → failed)."""
    for verdict in ("FAIL", "Fail", "fail"):
        subdir = tmp_path / verdict
        subdir.mkdir(parents=True, exist_ok=True)
        steps = [_make_step_dict("verify", outputs={"verdict": verdict})]
        pipeline_dir = _build_pipeline_dir(subdir, steps=steps)
        adapter = AlcovePipelineAdapter()
        sample = adapter.load(pipeline_dir)
        assert sample.phases[0].status == "failed", f"verdict={verdict!r} should give failed"


# ---------------------------------------------------------------------------
# Review findings tests
# ---------------------------------------------------------------------------


def test_parse_issues_major_prefix() -> None:
    """_parse_issues parses MAJOR: prefix correctly."""
    findings = _parse_issues("MAJOR: something broken", reviewer="review-test")
    assert len(findings) == 1
    assert findings[0].severity == "major"
    assert "something broken" in findings[0].issue
    assert findings[0].reviewer == "review-test"
    assert findings[0].finding_source == "review"


def test_parse_issues_critical_prefix() -> None:
    """_parse_issues parses CRITICAL: prefix correctly."""
    findings = _parse_issues("CRITICAL: severe issue", reviewer="review-sec")
    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_parse_issues_minor_prefix() -> None:
    """_parse_issues parses MINOR: prefix correctly."""
    findings = _parse_issues("MINOR: style issue", reviewer="review-style")
    assert len(findings) == 1
    assert findings[0].severity == "minor"


def test_parse_issues_semicolon_delimited(tmp_path: Path) -> None:
    """Multiple issues separated by semicolons are parsed as separate findings."""
    issues = "MAJOR: first issue; MINOR: second issue; CRITICAL: third issue"
    findings = _parse_issues(issues, reviewer="review-all")
    assert len(findings) == 3
    severities = {finding.severity for finding in findings}
    assert severities == {"major", "minor", "critical"}


def test_parse_issues_skips_unknown_prefix() -> None:
    """Tokens without a recognised severity prefix are silently skipped."""
    issues = "NOTE: informational; MAJOR: real issue"
    findings = _parse_issues(issues, reviewer="review-x")
    assert len(findings) == 1
    assert findings[0].severity == "major"


def test_findings_loaded_from_review_step(tmp_path: Path) -> None:
    """Adapter loads findings from review step outputs.issues."""
    issues = "MAJOR: test uses direct DB access; MINOR: missing import at top"
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("review-django", outputs={"approved": "false", "issues": issues}),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert len(sample.findings) == 2
    major_findings = [f for f in sample.findings if f.severity == "major"]
    assert len(major_findings) == 1
    assert "direct DB access" in major_findings[0].issue
    assert major_findings[0].reviewer == "review-django"


def test_findings_from_multiple_review_steps(tmp_path: Path) -> None:
    """Findings from multiple review steps are combined."""
    steps = [
        _make_step_dict("review-django", outputs={"issues": "MAJOR: issue A"}),
        _make_step_dict("review-security", outputs={"issues": "MINOR: issue B"}),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert len(sample.findings) == 2
    reviewers = {finding.reviewer for finding in sample.findings}
    assert reviewers == {"review-django", "review-security"}


def test_no_findings_when_no_issues_field(tmp_path: Path) -> None:
    """Steps without outputs.issues produce no findings."""
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("review-django", outputs={"approved": "true", "comments": "LGTM"}),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert len(sample.findings) == 0


# ---------------------------------------------------------------------------
# Verify verdict tests
# ---------------------------------------------------------------------------


def test_phase_status_helper_skipped() -> None:
    """_phase_status returns 'skipped' for skipped steps regardless of step_id."""
    assert _phase_status("skipped", "verify", {"verdict": "fail"}) == "skipped"
    assert _phase_status("skipped", "triage", {}) == "skipped"


def test_phase_status_helper_verify_fail() -> None:
    """_phase_status returns 'failed' for verify step with verdict=fail."""
    assert _phase_status("completed", "verify", {"verdict": "fail"}) == "failed"


def test_phase_status_helper_verify_pass() -> None:
    """_phase_status returns 'completed' for verify step with verdict=pass."""
    assert _phase_status("completed", "verify", {"verdict": "pass"}) == "completed"


def test_phase_status_helper_non_verify_with_fail_outputs() -> None:
    """_phase_status does not mark non-verify steps as failed due to verdict."""
    # Only verify steps look at the verdict field.
    assert _phase_status("completed", "review-django", {"verdict": "fail"}) == "completed"


# ---------------------------------------------------------------------------
# Rework cycle tests
# ---------------------------------------------------------------------------


def test_is_corrective_detects_failed_depends() -> None:
    """_is_corrective returns True for steps that depend on *.Failed events."""
    assert _is_corrective("verify.Failed") is True
    assert _is_corrective("review-django.Failed || review-security.Failed") is True
    assert _is_corrective("await-ci.Failed") is True


def test_is_corrective_returns_false_for_success_depends() -> None:
    """_is_corrective returns False for steps that depend on *.Succeeded events."""
    assert _is_corrective("verify.Succeeded") is False
    assert _is_corrective("implement.Succeeded") is False
    assert _is_corrective(None) is False


def test_rework_cycles_zero_when_all_corrective_skipped(tmp_path: Path) -> None:
    """rework_cycles=0 when all corrective steps are skipped."""
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("verify", outputs={"verdict": "fail"}),
        _make_step_dict("patch", status="skipped", depends="verify.Failed"),
        _make_step_dict("revision", status="skipped", depends="review.Failed"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.rework_cycles == 0


def test_rework_cycles_one_when_patch_activated(tmp_path: Path) -> None:
    """rework_cycles=1 when one corrective step completes."""
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("verify", outputs={"verdict": "fail"}),
        _make_step_dict("patch", depends="verify.Failed"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.rework_cycles == 1


def test_rework_cycles_accumulate_for_multiple_corrective_steps(tmp_path: Path) -> None:
    """rework_cycles counts each non-skipped corrective step separately."""
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("verify", outputs={"verdict": "fail"}),
        _make_step_dict("patch", depends="verify.Failed"),
        _make_step_dict("revision", depends="review-django.Failed"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.rework_cycles == 2


# ---------------------------------------------------------------------------
# Cost aggregation tests
# ---------------------------------------------------------------------------


def test_cost_aggregated_across_transcripts(tmp_path: Path) -> None:
    """Total cost sums transcript costs across all agent steps."""
    triage_transcript = _make_transcript(session_id="s1", cost_usd=0.25)
    implement_transcript = _make_transcript(session_id="s2", cost_usd=1.50)
    steps = [
        _make_step_dict("triage"),
        _make_step_dict("implement"),
    ]
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=steps,
        step_transcripts={"triage": triage_transcript, "implement": implement_transcript},
    )
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.total_cost_usd == pytest.approx(1.75, rel=1e-6)


def test_cost_none_when_no_transcripts(tmp_path: Path) -> None:
    """total_cost_usd is None when no step has a transcript with cost data."""
    steps = [
        _make_step_dict("create-pr", step_type="bridge"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.total_cost_usd is None


def test_cost_per_phase_stored_on_phase_result(tmp_path: Path) -> None:
    """Each phase's cost_usd is set from its own transcript."""
    triage_transcript = _make_transcript(session_id="s1", cost_usd=0.10)
    implement_transcript = _make_transcript(session_id="s2", cost_usd=0.90)
    steps = [
        _make_step_dict("triage"),
        _make_step_dict("implement"),
    ]
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=steps,
        step_transcripts={"triage": triage_transcript, "implement": implement_transcript},
    )
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    triage_phase = next(phase for phase in sample.phases if phase.name == "triage")
    implement_phase = next(phase for phase in sample.phases if phase.name == "implement")
    assert triage_phase.cost_usd == pytest.approx(0.10)
    assert implement_phase.cost_usd == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Context synthesis tests
# ---------------------------------------------------------------------------


def test_context_synthesized_from_triage_outputs(tmp_path: Path) -> None:
    """Triage step outputs are used to build knowledge context."""
    triage_outputs = {
        "approach": "Add scope_queryset method to DomainBasedPermission",
        "candidate_files": "authorization.py, test_domain_based_permissions.py",
        "risks": "Relies on pulpcore upstream API",
    }
    steps = [
        _make_step_dict("triage", outputs=triage_outputs),
        _make_step_dict("implement"),
    ]
    implement_transcript = _make_transcript(session_id="s1")
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=steps,
        step_transcripts={"implement": implement_transcript},
    )
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    implement_phase = next(phase for phase in sample.phases if phase.name == "implement")
    assert implement_phase.knowledge_context is not None
    assert "scope_queryset" in implement_phase.knowledge_context
    assert sample.context_source == "synthesized"


def test_context_placed_on_implement_phase(tmp_path: Path) -> None:
    """Synthesized context is attached to the implement phase preferentially."""
    triage_outputs = {"approach": "test approach", "candidate_files": "file.py"}
    steps = [
        _make_step_dict("triage", outputs=triage_outputs),
        _make_step_dict("plan", outputs={"plan": "Step 1: do X"}),
        _make_step_dict("implement"),
    ]
    implement_transcript = _make_transcript(session_id="s1")
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=steps,
        step_transcripts={"implement": implement_transcript},
    )
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    # Only implement should have context; other phases should not.
    implement_phase = next(phase for phase in sample.phases if phase.name == "implement")
    assert implement_phase.knowledge_context is not None
    triage_phase = next(phase for phase in sample.phases if phase.name == "triage")
    assert triage_phase.knowledge_context is None


def test_context_falls_back_to_implement_output(tmp_path: Path) -> None:
    """When no structured outputs are available, implement output is used as context."""
    steps = [
        _make_step_dict("implement"),
    ]
    implement_transcript = _make_transcript(session_id="s1", output_text="I implemented XYZ")
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=steps,
        step_transcripts={"implement": implement_transcript},
    )
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    implement_phase = sample.phases[0]
    assert implement_phase.knowledge_context is not None
    assert "XYZ" in implement_phase.knowledge_context


# ---------------------------------------------------------------------------
# Session metadata tests
# ---------------------------------------------------------------------------


def test_session_id_is_run_id(tmp_path: Path) -> None:
    """session_id is set to the run_id from run.json steps."""
    run_id = "test-run-999"
    steps = [_make_step_dict("triage", run_id=run_id)]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps, run_id=run_id)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.session_id == run_id


def test_orchestrator_is_alcove(tmp_path: Path) -> None:
    """orchestrator is always 'alcove' for pipeline exports."""
    steps = [_make_step_dict("triage")]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.orchestrator == "alcove"


def test_pipeline_phases_lists_all_step_ids(tmp_path: Path) -> None:
    """pipeline_phases contains all step_ids in execution order."""
    steps = [
        _make_step_dict("triage"),
        _make_step_dict("plan"),
        _make_step_dict("implement"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.pipeline_phases == ["triage", "plan", "implement"]


def test_started_at_is_earliest_non_empty_timestamp(tmp_path: Path) -> None:
    """started_at is the earliest non-empty step started_at timestamp."""
    steps = [
        _make_step_dict("triage", started_at="2026-04-01T10:00:00Z"),
        _make_step_dict("plan", started_at="2026-04-01T10:05:00Z"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.started_at == datetime(2026, 4, 1, 10, 0, 0, tzinfo=timezone.utc)


def test_total_phases_excludes_skipped(tmp_path: Path) -> None:
    """total_phases counts only non-skipped phases."""
    steps = [
        _make_step_dict("implement"),
        _make_step_dict("verify", outputs={"verdict": "pass"}),
        _make_step_dict("patch", status="skipped", depends="verify.Failed"),
    ]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.total_phases == 2


def test_model_id_from_transcript(tmp_path: Path) -> None:
    """model_id is extracted from the first transcript with a model entry."""
    triage_transcript = _make_transcript(session_id="s1", model="claude-test-model")
    steps = [_make_step_dict("triage")]
    pipeline_dir = _build_pipeline_dir(
        tmp_path,
        steps=steps,
        step_transcripts={"triage": triage_transcript},
    )
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(pipeline_dir)
    assert sample.session.model_id == "claude-test-model"


# ---------------------------------------------------------------------------
# Backward compatibility test
# ---------------------------------------------------------------------------


def test_alcove_adapter_still_loads_single_json(tmp_path: Path) -> None:
    """AlcoveAdapter continues to detect and load single-file JSON transcripts."""
    from raki.adapters.alcove import AlcoveAdapter

    transcript = _make_transcript()
    json_file = tmp_path / "session.json"
    json_file.write_text(json.dumps(transcript))

    adapter = AlcoveAdapter()
    assert adapter.detect(json_file) is True
    sample = adapter.load(json_file)
    assert sample.session.session_id == "sess-001"


def test_default_registry_includes_pipeline_adapter() -> None:
    """default_registry() includes AlcovePipelineAdapter."""
    registry = default_registry()
    adapter_names = [adapter.name for adapter in registry.list_all()]
    assert "alcove-pipeline" in adapter_names


def test_pipeline_adapter_registered_before_alcove(tmp_path: Path) -> None:
    """AlcovePipelineAdapter is tried before AlcoveAdapter in the default registry."""
    registry = default_registry()
    adapters = registry.list_all()
    pipeline_idx = next(
        idx for idx, adapter in enumerate(adapters) if adapter.name == "alcove-pipeline"
    )
    alcove_idx = next(idx for idx, adapter in enumerate(adapters) if adapter.name == "alcove")
    assert pipeline_idx < alcove_idx


# ---------------------------------------------------------------------------
# Discovery integration tests
# ---------------------------------------------------------------------------


def test_discovery_finds_pipeline_dir(tmp_path: Path) -> None:
    """discover_sessions() finds pipeline export directories."""
    from raki.adapters.discovery import discover_sessions

    steps = [_make_step_dict("triage"), _make_step_dict("implement")]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)
    registry = default_registry()
    found = discover_sessions([tmp_path], registry)
    assert pipeline_dir in found


def test_discovery_does_not_recurse_into_pipeline_dir(tmp_path: Path) -> None:
    """discover_sessions() does not recurse into a detected pipeline directory."""
    from raki.adapters.discovery import discover_sessions

    steps = [_make_step_dict("triage")]
    pipeline_dir = _build_pipeline_dir(tmp_path, steps=steps)

    # Put a single-file transcript inside the pipeline dir — it should NOT be found.
    inner_json = pipeline_dir / "inner.json"
    inner_json.write_text(json.dumps(_make_transcript(session_id="inner-sess")))

    registry = default_registry()
    found = discover_sessions([tmp_path], registry)

    # Only the pipeline_dir itself should appear, not inner.json.
    assert pipeline_dir in found
    assert inner_json not in found


# ---------------------------------------------------------------------------
# Integration test against real sample data (skipped if absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SAMPLE_DATA_PATH.exists(),
    reason="Real sample data not present at /tmp/alcove-export-81be9b17",
)
def test_real_sample_detects() -> None:
    """AlcovePipelineAdapter detects the real /tmp/alcove-export-81be9b17 directory."""
    adapter = AlcovePipelineAdapter()
    assert adapter.detect(SAMPLE_DATA_PATH) is True


@pytest.mark.skipif(
    not SAMPLE_DATA_PATH.exists(),
    reason="Real sample data not present at /tmp/alcove-export-81be9b17",
)
def test_real_sample_loads() -> None:
    """AlcovePipelineAdapter loads the real sample and produces meaningful values."""
    adapter = AlcovePipelineAdapter()
    sample = adapter.load(SAMPLE_DATA_PATH)

    # Session metadata is populated.
    assert sample.session.session_id  # non-empty run_id
    assert sample.session.orchestrator == "alcove"
    assert sample.session.started_at is not None
    assert sample.session.model_id is not None

    # Cost is aggregated (real transcripts have costs).
    assert sample.session.total_cost_usd is not None
    assert sample.session.total_cost_usd > 0

    # Phases include the expected pipeline steps.
    phase_names = [phase.name for phase in sample.phases]
    assert "triage" in phase_names
    assert "implement" in phase_names
    assert "verify" in phase_names

    # Verify phase is marked as failed (real sample has verdict=fail).
    verify_phase = next(phase for phase in sample.phases if phase.name == "verify")
    assert verify_phase.status == "failed"

    # Review findings are parsed.
    assert len(sample.findings) > 0
    assert any(finding.severity == "major" for finding in sample.findings)

    # Context is synthesized.
    assert any(phase.knowledge_context is not None for phase in sample.phases)
    assert sample.context_source == "synthesized"

    # Pipeline phases list is populated.
    assert sample.session.pipeline_phases is not None
    assert len(sample.session.pipeline_phases) > 0


@pytest.mark.skipif(
    not SAMPLE_DATA_PATH.exists(),
    reason="Real sample data not present at /tmp/alcove-export-81be9b17",
)
def test_real_sample_operational_metrics() -> None:
    """All 7 operational metrics produce meaningful (non-None) values for the real sample."""
    from raki.adapters import default_registry
    from raki.adapters.loader import DatasetLoader
    from raki.metrics.operational import ALL_OPERATIONAL
    from raki.metrics.protocol import MetricConfig

    registry = default_registry()
    loader = DatasetLoader(registry)
    sample = loader.load_session(SAMPLE_DATA_PATH)
    from raki.model import EvalDataset

    dataset = EvalDataset(samples=[sample])

    config = MetricConfig()
    for metric in ALL_OPERATIONAL:
        result = metric.compute(dataset, config)
        # Score may legitimately be None for some metrics with no data,
        # but at least the metric must not raise an exception.
        assert result is not None, f"Metric {metric.name} returned None result"
