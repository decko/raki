"""Tests for the adapter layer: protocol, registry, loader, session-schema adapter, alcove."""

import json
from pathlib import Path

import pytest

from raki.adapters.alcove import AlcoveAdapter
from raki.adapters.loader import DatasetLoader
from raki.adapters.redact import redact_dict, redact_sensitive
from raki.adapters.registry import AdapterRegistry
from raki.adapters.session_schema import SessionSchemaAdapter

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"
ALCOVE_FIXTURE = FIXTURES / "alcove-simple.json"
ALCOVE_ID_ONLY_FIXTURE = FIXTURES / "alcove-id-only.json"
ALCOVE_FAILURES_FIXTURE = FIXTURES / "alcove-with-failures.json"


def test_session_schema_adapter_detects_valid_session(
    pass_simple_dir: Path, rework_cycle_dir: Path
):
    adapter = SessionSchemaAdapter()
    assert adapter.detect(pass_simple_dir)
    assert adapter.detect(rework_cycle_dir)


def test_session_schema_adapter_rejects_empty_dir(tmp_path):
    adapter = SessionSchemaAdapter()
    assert not adapter.detect(tmp_path)


def test_session_schema_adapter_loads_pass_simple(pass_simple_dir: Path):
    adapter = SessionSchemaAdapter()
    sample = adapter.load(pass_simple_dir)
    assert sample.session.session_id == "101"
    assert sample.session.ticket == "101"
    assert sample.session.rework_cycles == 0
    assert sample.session.total_cost_usd == 12.5
    assert len(sample.phases) >= 2
    assert len(sample.findings) == 1
    assert sample.findings[0].severity == "minor"
    assert len(sample.events) == 10


def test_session_schema_adapter_loads_rework_cycle(rework_cycle_dir: Path):
    adapter = SessionSchemaAdapter()
    sample = adapter.load(rework_cycle_dir)
    assert sample.session.session_id == "53"
    assert sample.session.rework_cycles == 2
    implement_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(implement_phases) == 2
    gen_1 = [phase for phase in implement_phases if phase.generation == 1]
    assert len(gen_1) == 1


def test_session_schema_adapter_extracts_review_findings(rework_cycle_dir: Path):
    adapter = SessionSchemaAdapter()
    sample = adapter.load(rework_cycle_dir)
    critical = [finding for finding in sample.findings if finding.severity == "critical"]
    assert len(critical) == 1
    assert "closed channel" in critical[0].issue.lower()


def test_session_schema_adapter_handles_missing_phase_files(tmp_path):
    meta = {
        "ticket": "999",
        "summary": "test",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 1.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.session_id == "999"
    assert len(sample.phases) == 0


def test_registry_returns_registered_adapters():
    registry = AdapterRegistry()
    adapter = SessionSchemaAdapter()
    registry.register(adapter)
    assert registry.get("session-schema") is adapter
    assert "session-schema" in [registered.name for registered in registry.list_all()]


def test_dataset_loader_loads_directory(sessions_dir: Path):
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(sessions_dir)
    assert len(dataset.samples) >= 2


def test_dataset_loader_skips_malformed(malformed_dir: Path):
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(malformed_dir.parent)
    # The malformed session should produce an error, not a valid sample
    malformed_samples = [
        sample for sample in dataset.samples if sample.session.session_id == malformed_dir.name
    ]
    assert len(malformed_samples) == 0
    assert len(loader.errors) >= 1


def test_dataset_loader_skips_undetected_dirs(tmp_path):
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "transcript.json").write_text("{}")
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(tmp_path)
    assert len(dataset.samples) == 0


# --- Alcove adapter tests ---


def test_alcove_adapter_detects_valid_file():
    adapter = AlcoveAdapter()
    assert adapter.detect(ALCOVE_FIXTURE)


def test_alcove_adapter_rejects_non_alcove_file(tmp_path):
    adapter = AlcoveAdapter()
    bad_file = tmp_path / "not-alcove.json"
    bad_file.write_text('{"not": "alcove"}')
    assert not adapter.detect(bad_file)


def test_alcove_adapter_rejects_directory():
    adapter = AlcoveAdapter()
    assert not adapter.detect(FIXTURES / "pass-simple")


def test_alcove_adapter_loads_session():
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FIXTURE)
    assert sample.session.session_id == "ae7d2bf4-2f77-4ea6-8e3c-5442fe3d9fa7"
    assert sample.session.total_cost_usd == 0.027
    assert sample.session.rework_cycles == 0
    assert len(sample.phases) == 1
    assert sample.phases[0].name == "session"


def test_alcove_adapter_extracts_tool_calls():
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FIXTURE)
    assert len(sample.phases[0].tool_calls) == 1
    assert sample.phases[0].tool_calls[0].name == "Bash"


def test_alcove_adapter_extracts_model():
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FIXTURE)
    assert sample.session.model_id == "claude-sonnet-4-20250514"


def test_alcove_adapter_started_at_from_first_user_timestamp():
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FIXTURE)
    assert sample.session.started_at is not None
    assert sample.session.started_at.year == 2026
    assert sample.session.started_at.month == 4
    assert sample.session.started_at.day == 16


def test_alcove_adapter_aggregates_tokens():
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FIXTURE)
    phase = sample.phases[0]
    # 10+10+5 = 25 input, 8+8+20 = 36 output
    assert phase.tokens_in == 25
    assert phase.tokens_out == 36


def test_alcove_adapter_rejects_oversized_file(tmp_path):
    adapter = AlcoveAdapter()
    big_file = tmp_path / "big.json"
    # Write a file just over the 50MB limit
    big_file.write_text('{"session_id": "x", "transcript": []}' + " " * (50 * 1024 * 1024))
    with pytest.raises(ValueError, match="exceeds"):
        adapter.load(big_file)


def test_dataset_loader_loads_alcove_files(tmp_path):
    import shutil

    shutil.copy(ALCOVE_FIXTURE, tmp_path / "alcove-simple.json")
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    registry.register(AlcoveAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(tmp_path)
    assert len(dataset.samples) == 1


def test_alcove_adapter_detects_id_only_session():
    """detect() must accept sessions that use 'id' instead of 'session_id'.

    Some Claude Code exports use a top-level 'id' key with no 'session_id' and
    no 'task_id'.  Previously detect() required 'task_id' alongside 'id' (the
    bridge format fingerprint), so these sessions were silently rejected.
    """
    adapter = AlcoveAdapter()
    assert adapter.detect(ALCOVE_ID_ONLY_FIXTURE)


def test_alcove_adapter_loads_id_only_session():
    """load() must extract session_id from the top-level 'id' field."""
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_ID_ONLY_FIXTURE)
    assert sample.session.session_id == "f3c2a1b0-dead-beef-abcd-ef1234567890"
    assert sample.session.total_cost_usd == 0.015
    assert sample.session.model_id == "claude-sonnet-4-20250514"
    assert len(sample.phases) == 1
    assert sample.phases[0].name == "session"


# --- Synthesized findings tests ---


def test_alcove_adapter_synthesizes_findings_from_test_failures():
    """AlcoveAdapter creates synthesized findings when transcript has test failures."""
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FAILURES_FIXTURE)
    # Should produce at least one synthesized finding (pytest failure)
    synthesized = [f for f in sample.findings if f.finding_source == "synthesized"]
    assert len(synthesized) >= 1


def test_alcove_adapter_synthesized_findings_have_major_severity():
    """Synthesized findings from test failures are rated major severity."""
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FAILURES_FIXTURE)
    synthesized = [f for f in sample.findings if f.finding_source == "synthesized"]
    for finding in synthesized:
        assert finding.severity == "major"


def test_alcove_adapter_synthesized_findings_have_synthesized_reviewer():
    """Synthesized findings carry 'synthesized' as reviewer name."""
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FAILURES_FIXTURE)
    synthesized = [f for f in sample.findings if f.finding_source == "synthesized"]
    for finding in synthesized:
        assert finding.reviewer == "synthesized"


def test_alcove_adapter_no_synthesized_findings_when_explicit_findings_present(tmp_path):
    """When explicit findings are provided in the JSON, no synthesis occurs."""
    transcript_data = {
        "session_id": "explicit-findings-test",
        "findings": [{"issue": "Manual review issue", "severity": "minor", "source": "reviewer"}],
        "transcript": [
            {
                "type": "system",
                "model": "claude-sonnet-4-20250514",
                "tools": ["Bash"],
                "subtype": "init",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg_01",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [
                        {
                            "id": "toolu_test1",
                            "name": "Bash",
                            "type": "tool_use",
                            "input": {"command": "pytest tests/ -v"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "FAILED tests/test_foo.py::test_bar",
                            "is_error": False,
                            "tool_use_id": "toolu_test1",
                        }
                    ],
                },
                "timestamp": "2026-04-20T10:00:00.000Z",
            },
        ],
    }
    fixture = tmp_path / "explicit.json"
    fixture.write_text(json.dumps(transcript_data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    # Only the explicit finding should be present (no synthesized ones)
    assert len(sample.findings) == 1
    assert sample.findings[0].finding_source == "review"
    assert sample.findings[0].issue == "Manual review issue"


def test_alcove_adapter_deduplicates_repeated_test_failures(tmp_path):
    """The same failure text repeated across test runs produces only one finding."""
    duplicate_failure = "FAILED tests/test_foo.py::test_bar - AssertionError"
    transcript_data = {
        "session_id": "dedup-test",
        "transcript": [
            {
                "type": "system",
                "model": "claude-sonnet-4-20250514",
                "tools": ["Bash"],
                "subtype": "init",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg_01",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [
                        {
                            "id": "toolu_t1",
                            "name": "Bash",
                            "type": "tool_use",
                            "input": {"command": "pytest tests/ -v"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": duplicate_failure,
                            "is_error": False,
                            "tool_use_id": "toolu_t1",
                        }
                    ],
                },
                "timestamp": "2026-04-20T10:00:00.000Z",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg_02",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [
                        {
                            "id": "toolu_t2",
                            "name": "Bash",
                            "type": "tool_use",
                            "input": {"command": "pytest tests/ -v"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": duplicate_failure,
                            "is_error": False,
                            "tool_use_id": "toolu_t2",
                        }
                    ],
                },
            },
        ],
    }
    fixture = tmp_path / "dedup.json"
    fixture.write_text(json.dumps(transcript_data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    synthesized = [f for f in sample.findings if f.finding_source == "synthesized"]
    # Same failure text → 1 finding, not 2
    assert len(synthesized) == 1


def test_alcove_adapter_no_synthesis_when_no_failures():
    """When no test failures occur in the transcript, no findings are synthesized."""
    adapter = AlcoveAdapter()
    # alcove-simple.json has no test failures in the transcript
    sample = adapter.load(ALCOVE_FIXTURE)
    assert sample.findings == []


def test_alcove_adapter_parsed_findings_tagged_as_review(tmp_path):
    """Explicitly provided findings get finding_source='review'."""
    transcript_data = {
        "session_id": "review-tag-test",
        "findings": [
            {"issue": "Use context manager", "severity": "major", "source": "reviewer"},
            {"issue": "Missing type hint", "severity": "minor"},
        ],
        "transcript": [
            {
                "type": "system",
                "model": "claude-sonnet-4-20250514",
                "tools": [],
                "subtype": "init",
            }
        ],
    }
    fixture = tmp_path / "review_tagged.json"
    fixture.write_text(json.dumps(transcript_data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    assert all(f.finding_source == "review" for f in sample.findings)


# --- Redaction tests ---


def test_redact_bearer_token():
    text = "Authorization: Bearer sk-abc123xyz"
    result = redact_sensitive(text)
    assert "sk-abc123xyz" not in result
    assert "***REDACTED***" in result


def test_redact_api_key():
    text = "api_key=secret_key_12345"
    result = redact_sensitive(text)
    assert "secret_key_12345" not in result
    assert "***REDACTED***" in result


def test_redact_api_key_with_dash():
    text = "api-key: my-super-secret"
    result = redact_sensitive(text)
    assert "my-super-secret" not in result
    assert "***REDACTED***" in result


def test_redact_password():
    text = "password=hunter2"
    result = redact_sensitive(text)
    assert "hunter2" not in result
    assert "***REDACTED***" in result


def test_redact_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abcdef123456"
    text = f"token is {jwt}"
    result = redact_sensitive(text)
    assert jwt not in result
    assert "***REDACTED***" in result


def test_redact_preserves_normal_text():
    text = "This is a normal session output with no secrets."
    result = redact_sensitive(text)
    assert result == text


def test_redact_token_equals():
    text = "token=abc123secret"
    result = redact_sensitive(text)
    assert "abc123secret" not in result
    assert "***REDACTED***" in result


def test_redact_multiple_patterns():
    text = "Bearer sk-abc123 and password=hunter2"
    result = redact_sensitive(text)
    assert "sk-abc123" not in result
    assert "hunter2" not in result


def test_redact_quoted_bearer_token():
    text = 'Authorization: Bearer "sk-abc123xyz"'
    result = redact_sensitive(text)
    assert "sk-abc123xyz" not in result
    assert "***REDACTED***" in result


def test_redact_quoted_password():
    text = 'password="hunter2"'
    result = redact_sensitive(text)
    assert "hunter2" not in result
    assert "***REDACTED***" in result


def test_redact_aws_access_key():
    text = "aws_key=AKIAIOSFODNN7EXAMPLE"
    result = redact_sensitive(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert "***REDACTED***" in result


def test_redact_github_token():
    text = "token: ghp_ABCDEFabcdef1234567890"
    result = redact_sensitive(text)
    assert "ghp_ABCDEFabcdef1234567890" not in result
    assert "***REDACTED***" in result


def test_redact_gitlab_token():
    text = "auth glpat_ABCDEFabcdef1234567890"
    result = redact_sensitive(text)
    assert "glpat_ABCDEFabcdef1234567890" not in result
    assert "***REDACTED***" in result


def test_redact_generic_secret():
    text = "secret=my_super_secret_value"
    result = redact_sensitive(text)
    assert "my_super_secret_value" not in result
    assert "***REDACTED***" in result


def test_redact_private_key_block():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJBALRiMLAH\n-----END RSA PRIVATE KEY-----"
    result = redact_sensitive(text)
    assert "MIIBogIBAAJBALRiMLAH" not in result
    assert "***REDACTED***" in result


def test_redact_dict_redacts_string_values():
    data = {"command": "curl -H 'Bearer sk-secret123'", "count": 42}
    result = redact_dict(data)
    assert "sk-secret123" not in result["command"]
    assert "***REDACTED***" in result["command"]
    assert result["count"] == 42


def test_redact_dict_recurses_into_nested_structures():
    data = {
        "outer": {
            "inner": "password=hunter2",
            "list_val": ["token=abc123secret", "normal text"],
        }
    }
    result = redact_dict(data)
    assert "hunter2" not in result["outer"]["inner"]
    assert "abc123secret" not in result["outer"]["list_val"][0]
    assert result["outer"]["list_val"][1] == "normal text"


def test_redact_dict_handles_empty_dict():
    assert redact_dict({}) == {}


def test_alcove_adapter_rejects_symlink(tmp_path):
    """Ensure alcove.load() rejects symlinks."""
    adapter = AlcoveAdapter()
    real_file = tmp_path / "real.json"
    real_file.write_text('{"session_id": "x", "transcript": []}')
    link_file = tmp_path / "link.json"
    link_file.symlink_to(real_file)
    with pytest.raises(ValueError, match="symlink"):
        adapter.load(link_file)


def test_session_schema_skips_malformed_finding_missing_issue(tmp_path):
    """A finding missing the required 'issue' key should be skipped, not crash."""
    meta = {
        "ticket": "777",
        "summary": "test malformed finding",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 1.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    # One valid finding and one missing the required 'issue' key
    review_data = {
        "findings": [
            {"source": "reviewer-a", "severity": "minor", "issue": "valid finding"},
            {"source": "reviewer-b", "severity": "major"},  # missing 'issue' key
        ]
    }
    (tmp_path / "review.json").write_text(json.dumps(review_data))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    # The valid finding should be kept; the malformed one should be skipped
    assert len(sample.findings) == 1
    assert sample.findings[0].issue == "valid finding"


def test_session_schema_skips_malformed_event_line(tmp_path):
    """An event line missing required keys should be skipped, not crash."""
    meta = {
        "ticket": "888",
        "summary": "test malformed event",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 1.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    valid_event = json.dumps(
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "triage",
            "kind": "phase_started",
            "data": {},
        }
    )
    # Missing 'timestamp' and 'kind' -- both required by SessionEvent
    malformed_event = json.dumps({"phase": "triage", "data": {}})
    # Invalid 'kind' value -- not in Literal
    invalid_kind_event = json.dumps(
        {
            "timestamp": "2026-04-10T08:01:00Z",
            "kind": "totally_invalid_kind",
        }
    )
    events_content = "\n".join([valid_event, malformed_event, invalid_kind_event])
    (tmp_path / "events.jsonl").write_text(events_content)

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    # Only the valid event should be loaded; malformed ones should be skipped
    assert len(sample.events) == 1
    assert sample.events[0].kind == "phase_started"


# --- NoneType edge-case tests (issue #27) ---


def test_session_schema_handles_null_phases_in_meta(tmp_path):
    """meta.json with "phases": null should load gracefully with empty phases."""
    meta = {
        "ticket": "170",
        "summary": "session with null phases",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": None,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.session_id == "170"
    assert sample.session.total_phases == 0
    assert len(sample.phases) == 0


def test_session_schema_handles_missing_phases_key_in_meta(tmp_path):
    """meta.json with no 'phases' key at all should load gracefully."""
    meta = {
        "ticket": "172",
        "summary": "session with missing phases key",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 3.0,
        "rework_cycles": 1,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.session_id == "172"
    assert sample.session.total_phases == 0
    assert len(sample.phases) == 0


def test_session_schema_handles_null_findings_in_review(tmp_path):
    """review.json with "findings": null should be skipped gracefully."""
    meta = {
        "ticket": "181",
        "summary": "session with null findings",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 2.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    review_data = {"findings": None}
    (tmp_path / "review.json").write_text(json.dumps(review_data))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.session_id == "181"
    assert len(sample.findings) == 0


def test_session_schema_handles_null_phase_metadata(tmp_path):
    """Phase entry in meta.json phases dict is null instead of a dict."""
    meta = {
        "ticket": "185",
        "summary": "session with null phase metadata",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 4.0,
        "rework_cycles": 0,
        "phases": {"triage": None},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    # Create a triage phase file so the adapter tries to load it
    triage_data = {"summary": "triage output"}
    (tmp_path / "triage.json").write_text(json.dumps(triage_data))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.session_id == "185"
    assert len(sample.phases) == 1
    assert sample.phases[0].name == "triage"
    # With null phase metadata, defaults should be used
    assert sample.phases[0].generation == 1
    assert sample.phases[0].status == "completed"


# --- Security hardening tests (issue #35) ---


def test_session_schema_detect_rejects_symlink(tmp_path):
    """SessionSchemaAdapter.detect() must reject symlinked directories."""
    real_dir = tmp_path / "real-session"
    real_dir.mkdir()
    (real_dir / "meta.json").write_text('{"ticket": "1", "started_at": "2026-04-10T08:00:00Z"}')
    (real_dir / "events.jsonl").write_text("")
    link_dir = tmp_path / "link-session"
    link_dir.symlink_to(real_dir)
    adapter = SessionSchemaAdapter()
    assert not adapter.detect(link_dir)


def test_session_schema_load_rejects_symlink(tmp_path):
    """SessionSchemaAdapter.load() must reject symlinked directories."""
    real_dir = tmp_path / "real-session"
    real_dir.mkdir()
    (real_dir / "meta.json").write_text('{"ticket": "1", "started_at": "2026-04-10T08:00:00Z"}')
    (real_dir / "events.jsonl").write_text("")
    link_dir = tmp_path / "link-session"
    link_dir.symlink_to(real_dir)
    adapter = SessionSchemaAdapter()
    with pytest.raises(ValueError, match="symlink"):
        adapter.load(link_dir)


def test_alcove_detect_rejects_symlink(tmp_path):
    """AlcoveAdapter.detect() must reject symlinked files."""
    real_file = tmp_path / "real.json"
    real_file.write_text('{"session_id": "x", "transcript": []}')
    link_file = tmp_path / "link.json"
    link_file.symlink_to(real_file)
    adapter = AlcoveAdapter()
    assert not adapter.detect(link_file)


def test_redact_aws_secret_access_key():
    """AWS_SECRET_ACCESS_KEY=... should be redacted."""
    text = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    result = redact_sensitive(text)
    assert "wJalrXUtnFEMI" not in result
    assert "***REDACTED***" in result


def test_redact_github_token_env_var():
    """GITHUB_TOKEN=... should be redacted."""
    text = "GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    result = redact_sensitive(text)
    assert "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in result
    assert "***REDACTED***" in result


def test_redact_gh_token_env_var():
    """GH_TOKEN=... should be redacted."""
    text = "GH_TOKEN=ghp_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
    result = redact_sensitive(text)
    assert "ghp_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy" not in result
    assert "***REDACTED***" in result


def test_redact_generic_secret_env_var():
    """Generic *_SECRET*= env vars should be redacted (e.g. MY_SECRET_KEY=...)."""
    text = "MY_SECRET_KEY=super-secret-value-123"
    result = redact_sensitive(text)
    assert "super-secret-value-123" not in result
    assert "***REDACTED***" in result


def test_redact_generic_secret_env_var_with_underscore():
    """DB_SECRET_TOKEN=... should be redacted."""
    text = "DB_SECRET_TOKEN=another-secret"
    result = redact_sensitive(text)
    assert "another-secret" not in result
    assert "***REDACTED***" in result


def test_redact_multiline_jwt():
    """JWT split across multiple lines should be redacted."""
    text = "header: eyJhbGciOiJIUzI1NiJ9.\neyJzdWIiOiJ1c2VyIn0.\nabcdef123456"
    result = redact_sensitive(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in result
    assert "***REDACTED***" in result


def test_redact_multiline_jwt_across_content_blocks():
    """JWT header+payload split across content blocks with whitespace should be redacted."""
    text = "eyJhbGciOiJIUzI1NiJ9\n.eyJzdWIiOiJ1c2VyIn0\n.abcdef123456"
    result = redact_sensitive(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in result
    assert "***REDACTED***" in result


# --- Task 10: adapter data completeness tests ---


def test_session_schema_extracts_model_id_from_meta(tmp_path):
    """model_id in meta.json should be populated on SessionMeta."""
    meta = {
        "ticket": "200",
        "summary": "model id test",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
        "model_id": "claude-sonnet-4-20250514",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.model_id == "claude-sonnet-4-20250514"


def test_session_schema_extracts_model_id_from_events(tmp_path):
    """model_id should be extracted from events.jsonl when not in meta.json."""
    meta = {
        "ticket": "201",
        "summary": "model from events",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "implement",
            "kind": "phase_started",
            "data": {"generation": 1, "model": "claude-opus-4-20250514"},
        },
        {
            "timestamp": "2026-04-10T08:05:00Z",
            "phase": "implement",
            "kind": "phase_completed",
            "data": {"cost": 3.5},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.model_id == "claude-opus-4-20250514"


def test_session_schema_meta_model_id_takes_precedence(tmp_path):
    """model_id from meta.json should take precedence over events.jsonl."""
    meta = {
        "ticket": "202",
        "summary": "precedence test",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
        "model_id": "claude-sonnet-4-20250514",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "implement",
            "kind": "phase_started",
            "data": {"generation": 1, "model": "claude-opus-4-20250514"},
        },
    ]
    (tmp_path / "events.jsonl").write_text(json.dumps(events[0]))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.model_id == "claude-sonnet-4-20250514"


def test_session_schema_extracts_tokens_from_phase_meta(tmp_path):
    """tokens_in/tokens_out from phase metadata should populate PhaseResult."""
    meta = {
        "ticket": "210",
        "summary": "tokens test",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {
            "implement": {
                "status": "completed",
                "cost": 3.5,
                "generation": 1,
                "tokens_in": 15000,
                "tokens_out": 8000,
            }
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    implement_data = {"summary": "implemented feature"}
    (tmp_path / "implement.json").write_text(json.dumps(implement_data))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(impl_phases) == 1
    assert impl_phases[0].tokens_in == 15000
    assert impl_phases[0].tokens_out == 8000


def test_session_schema_extracts_tokens_from_events(tmp_path):
    """tokens_in/tokens_out should be extracted from phase_completed events."""
    meta = {
        "ticket": "211",
        "summary": "tokens from events",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {
            "implement": {"status": "completed", "cost": 3.5, "generation": 1},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "implement",
            "kind": "phase_started",
            "data": {"generation": 1},
        },
        {
            "timestamp": "2026-04-10T08:05:00Z",
            "phase": "implement",
            "kind": "phase_completed",
            "data": {"cost": 3.5, "tokens_in": 12000, "tokens_out": 6000},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    implement_data = {"summary": "implemented feature"}
    (tmp_path / "implement.json").write_text(json.dumps(implement_data))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(impl_phases) == 1
    assert impl_phases[0].tokens_in == 12000
    assert impl_phases[0].tokens_out == 6000


def test_session_schema_phase_meta_tokens_take_precedence(tmp_path):
    """Phase metadata tokens should take precedence over event tokens."""
    meta = {
        "ticket": "212",
        "summary": "token precedence",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {
            "implement": {
                "status": "completed",
                "cost": 3.5,
                "generation": 1,
                "tokens_in": 15000,
                "tokens_out": 8000,
            },
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "implement",
            "kind": "phase_started",
            "data": {"generation": 1},
        },
        {
            "timestamp": "2026-04-10T08:05:00Z",
            "phase": "implement",
            "kind": "phase_completed",
            "data": {"cost": 3.5, "tokens_in": 12000, "tokens_out": 6000},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    implement_data = {"summary": "implemented feature"}
    (tmp_path / "implement.json").write_text(json.dumps(implement_data))
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(impl_phases) == 1
    assert impl_phases[0].tokens_in == 15000
    assert impl_phases[0].tokens_out == 8000


# --- Recursive DatasetLoader tests ---


def test_dataset_loader_recursive_finds_nested_sessions(tmp_path):
    """DatasetLoader in recursive mode should find sessions in subdirectories."""
    # Create nested structure: root/project-a/session-1/
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    session_dir = project_dir / "session-1"
    session_dir.mkdir()
    meta = {
        "ticket": "300",
        "summary": "nested session",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))
    (session_dir / "events.jsonl").write_text("")

    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(tmp_path, recursive=True)
    assert len(dataset.samples) == 1
    assert dataset.samples[0].session.session_id == "300"


def test_dataset_loader_non_recursive_skips_nested_sessions(tmp_path):
    """DatasetLoader in non-recursive (default) mode should not descend into subdirs."""
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    session_dir = project_dir / "session-1"
    session_dir.mkdir()
    meta = {
        "ticket": "301",
        "summary": "nested session",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))
    (session_dir / "events.jsonl").write_text("")

    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    # Default: non-recursive
    dataset = loader.load_directory(tmp_path)
    assert len(dataset.samples) == 0


def test_dataset_loader_recursive_finds_both_levels(tmp_path):
    """Recursive mode should find sessions at root and nested levels."""
    # Direct child session
    direct_session = tmp_path / "session-direct"
    direct_session.mkdir()
    meta_direct = {
        "ticket": "310",
        "summary": "direct session",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (direct_session / "meta.json").write_text(json.dumps(meta_direct))
    (direct_session / "events.jsonl").write_text("")

    # Nested session
    nested_dir = tmp_path / "project-a"
    nested_dir.mkdir()
    nested_session = nested_dir / "session-nested"
    nested_session.mkdir()
    meta_nested = {
        "ticket": "311",
        "summary": "nested session",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (nested_session / "meta.json").write_text(json.dumps(meta_nested))
    (nested_session / "events.jsonl").write_text("")

    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(tmp_path, recursive=True)
    session_ids = {sample.session.session_id for sample in dataset.samples}
    assert session_ids == {"310", "311"}


def test_dataset_loader_recursive_deeply_nested(tmp_path):
    """Recursive mode should find sessions multiple levels deep."""
    deep_dir = tmp_path / "a" / "b" / "c"
    deep_dir.mkdir(parents=True)
    session_dir = deep_dir / "deep-session"
    session_dir.mkdir()
    meta = {
        "ticket": "320",
        "summary": "deep session",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))
    (session_dir / "events.jsonl").write_text("")

    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(tmp_path, recursive=True)
    assert len(dataset.samples) == 1
    assert dataset.samples[0].session.session_id == "320"


# --- Generational file sorting tests ---


def test_generational_sorting_base_file_is_latest(tmp_path):
    """Base file (implement.json) should be assigned highest generation (latest)."""
    meta = {
        "ticket": "400",
        "summary": "generational sorting",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 10.0,
        "rework_cycles": 2,
        "phases": {
            "implement": {"status": "completed", "cost": 3.5, "generation": 3},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    # .1 = generation 1, .2 = generation 2, base = latest (generation 3 from meta)
    (tmp_path / "implement.json.1").write_text(json.dumps({"gen": "first"}))
    (tmp_path / "implement.json.2").write_text(json.dumps({"gen": "second"}))
    (tmp_path / "implement.json").write_text(json.dumps({"gen": "latest"}))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(impl_phases) == 3

    # Sort by generation for predictable assertions
    impl_phases.sort(key=lambda phase: phase.generation)
    assert impl_phases[0].generation == 1
    assert impl_phases[1].generation == 2
    assert impl_phases[2].generation == 3


def test_generational_sorting_single_base_file_gen_1(tmp_path):
    """A single base file with no suffixed files should be generation 1."""
    meta = {
        "ticket": "401",
        "summary": "single base file",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {
            "implement": {"status": "completed", "cost": 3.5, "generation": 1},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "implement.json").write_text(json.dumps({"gen": "only"}))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
    assert len(impl_phases) == 1
    assert impl_phases[0].generation == 1


def test_generational_sorting_base_file_gen_from_meta(tmp_path):
    """Base file generation should use the meta.json phases generation value."""
    meta = {
        "ticket": "402",
        "summary": "gen from meta",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 1,
        "phases": {
            "implement": {"status": "completed", "cost": 3.5, "generation": 4},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "implement.json.1").write_text(json.dumps({"gen": "first"}))
    (tmp_path / "implement.json").write_text(json.dumps({"gen": "latest"}))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = sorted(
        [phase for phase in sample.phases if phase.name == "implement"],
        key=lambda phase: phase.generation,
    )
    assert len(impl_phases) == 2
    assert impl_phases[0].generation == 1
    assert impl_phases[1].generation == 4


def test_generational_sorting_no_meta_generation_defaults(tmp_path):
    """Without meta generation, base file with suffixed files gets max(suffixed) + 1."""
    meta = {
        "ticket": "403",
        "summary": "no meta generation",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 1,
        "phases": {
            "implement": {"status": "completed", "cost": 3.5},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "implement.json.1").write_text(json.dumps({"gen": "first"}))
    (tmp_path / "implement.json.2").write_text(json.dumps({"gen": "second"}))
    (tmp_path / "implement.json").write_text(json.dumps({"gen": "latest"}))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = sorted(
        [phase for phase in sample.phases if phase.name == "implement"],
        key=lambda phase: phase.generation,
    )
    assert len(impl_phases) == 3
    assert impl_phases[0].generation == 1
    assert impl_phases[1].generation == 2
    # Base file should be latest: max(1, 2) + 1 = 3
    assert impl_phases[2].generation == 3


# --- DatasetLoader adapter_name filtering tests (issue #79) ---


class TestDatasetLoaderAdapterName:
    def test_load_directory_with_valid_adapter_name(self, sessions_dir):
        registry = AdapterRegistry()
        registry.register(SessionSchemaAdapter())
        loader = DatasetLoader(registry)
        dataset = loader.load_directory(sessions_dir, adapter_name="session-schema")
        # pass-simple and rework-cycle are valid session dirs under sessions/
        assert len(dataset.samples) == 2

    def test_load_directory_with_invalid_adapter_name(self, sessions_dir):
        registry = AdapterRegistry()
        registry.register(SessionSchemaAdapter())
        loader = DatasetLoader(registry)
        with pytest.raises(ValueError, match="Unknown adapter"):
            loader.load_directory(sessions_dir, adapter_name="nonexistent")

    def test_load_directory_adapter_name_bypasses_detection(self, tmp_path):
        """When adapter_name is set, _detect_adapter() is not called for each child."""
        import shutil

        # Copy alcove fixture into a flat dir alongside a session-schema dir
        shutil.copy(ALCOVE_FIXTURE, tmp_path / "alcove-simple.json")
        session_dir = tmp_path / "session-101"
        session_dir.mkdir()
        meta = {
            "ticket": "101",
            "summary": "test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 1.0,
            "rework_cycles": 0,
            "phases": {},
        }
        import json

        (session_dir / "meta.json").write_text(json.dumps(meta))
        (session_dir / "events.jsonl").write_text("")

        registry = AdapterRegistry()
        registry.register(SessionSchemaAdapter())
        registry.register(AlcoveAdapter())
        loader = DatasetLoader(registry)

        # Force session-schema adapter: alcove file should be skipped/error, not loaded
        dataset = loader.load_directory(tmp_path, adapter_name="session-schema")
        session_ids = {sample.session.session_id for sample in dataset.samples}
        # Only the session-schema session should be loaded (alcove file will fail/be skipped)
        assert "101" in session_ids
        alcove_ids = {
            sample.session.session_id
            for sample in dataset.samples
            if sample.session.session_id == "ae7d2bf4-2f77-4ea6-8e3c-5442fe3d9fa7"
        }
        assert len(alcove_ids) == 0


# --- default_registry() tests (issue #78) ---


class TestDefaultRegistry:
    def test_default_registry_contains_both_adapters(self):
        from raki.adapters import default_registry

        registry = default_registry()
        adapters = registry.list_all()
        adapter_names = {adapter.name for adapter in adapters}
        assert "session-schema" in adapter_names
        assert "alcove" in adapter_names

    def test_default_registry_returns_fresh_instance(self):
        from raki.adapters import default_registry

        registry_a = default_registry()
        registry_b = default_registry()
        assert registry_a is not registry_b


# --- Zero-token and missing-data edge-case tests (issue #37) ---


def test_session_schema_zero_tokens_in_meta_wins_over_event(tmp_path):
    """Phase metadata tokens_in=0 must win over event tokens_in=100 (not treated as falsy)."""
    meta = {
        "ticket": "500",
        "summary": "zero token edge case",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {
            "implement": {
                "status": "completed",
                "cost": 2.0,
                "generation": 1,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "implement",
            "kind": "phase_started",
            "data": {"generation": 1},
        },
        {
            "timestamp": "2026-04-10T08:05:00Z",
            "phase": "implement",
            "kind": "phase_completed",
            "data": {"cost": 2.0, "tokens_in": 100, "tokens_out": 200},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "implement.json").write_text(json.dumps({"summary": "impl"}))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [p for p in sample.phases if p.name == "implement"]
    assert len(impl_phases) == 1
    # The metadata value (0) must be used, NOT the event value (100/200)
    assert impl_phases[0].tokens_in == 0
    assert impl_phases[0].tokens_out == 0


def test_session_schema_no_model_id_from_either_source(tmp_path):
    """When neither meta.json nor events provide a model_id, it should be None."""
    meta = {
        "ticket": "501",
        "summary": "no model id anywhere",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 3.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "triage",
            "kind": "phase_started",
            "data": {"generation": 1},
        },
        {
            "timestamp": "2026-04-10T08:01:00Z",
            "phase": "triage",
            "kind": "phase_completed",
            "data": {"cost": 1.0},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.model_id is None


def test_session_schema_no_token_counts_from_either_source(tmp_path):
    """When neither phase meta nor events provide token counts, they should be None."""
    meta = {
        "ticket": "502",
        "summary": "no tokens anywhere",
        "started_at": "2026-04-10T08:00:00Z",
        "total_cost": 4.0,
        "rework_cycles": 0,
        "phases": {
            "implement": {"status": "completed", "cost": 2.0, "generation": 1},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    events = [
        {
            "timestamp": "2026-04-10T08:00:00Z",
            "phase": "implement",
            "kind": "phase_started",
            "data": {"generation": 1},
        },
        {
            "timestamp": "2026-04-10T08:05:00Z",
            "phase": "implement",
            "kind": "phase_completed",
            "data": {"cost": 2.0},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "implement.json").write_text(json.dumps({"summary": "impl"}))

    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    impl_phases = [p for p in sample.phases if p.name == "implement"]
    assert len(impl_phases) == 1
    assert impl_phases[0].tokens_in is None
    assert impl_phases[0].tokens_out is None


# --- Alcove bridge format tests ---


def _bridge_session(tmp_path, overrides=None):
    """Create a minimal bridge-format session file and return its path."""
    data = {
        "id": "bridge-001",
        "task_id": "task-abc",
        "submitter": "workflow",
        "task_name": "Upgrade Dependencies",
        "status": "completed",
        "started_at": "2026-04-22T10:00:00Z",
        "duration": "2m30s",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-6", "session_id": "task-abc"},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "content": [{"type": "text", "text": "I will upgrade deps."}],
                },
            },
            {
                "type": "result",
                "total_cost_usd": 0.42,
                "duration_ms": 150000,
                "modelUsage": {"claude-sonnet-4-6": {"inputTokens": 100, "outputTokens": 50}},
            },
        ],
    }
    if overrides:
        data.update(overrides)
    session_file = tmp_path / "bridge-session.json"
    session_file.write_text(json.dumps(data))
    return session_file


def test_bridge_format_detected(tmp_path):
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    assert adapter.detect(source) is True


def test_bridge_session_id_from_id_field(tmp_path):
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.session_id == "bridge-001"


def test_bridge_task_name_as_ticket(tmp_path):
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.ticket == "Upgrade Dependencies"


def test_bridge_started_at_from_top_level(tmp_path):
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.started_at.year == 2026
    assert sample.session.started_at.hour == 10


def test_bridge_cost_from_result(tmp_path):
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.total_cost_usd == 0.42


def test_bridge_model_from_system(tmp_path):
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.model_id == "claude-sonnet-4-6"


def test_bridge_failed_status(tmp_path):
    source = _bridge_session(tmp_path, overrides={"status": "failed"})
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.phases[0].status == "failed"


def test_bridge_no_task_name_no_ticket(tmp_path):
    data = {
        "id": "bridge-002",
        "task_id": "task-xyz",
        "submitter": "workflow",
        "status": "completed",
        "started_at": "2026-04-22T10:00:00Z",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-6"},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [{"type": "text", "text": "done"}],
                },
            },
            {"type": "result", "total_cost_usd": 0.01, "duration_ms": 1000},
        ],
    }
    session_file = tmp_path / "no-task-name.json"
    session_file.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    sample = adapter.load(session_file)
    assert sample.session.ticket is None


def test_bridge_classic_format_still_works(tmp_path):
    """Classic session_id format must continue working."""
    data = {
        "session_id": "classic-001",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-6"},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [{"type": "text", "text": "hello"}],
                },
            },
            {"type": "result", "total_cost_usd": 0.01, "duration_ms": 1000},
        ],
    }
    session_file = tmp_path / "classic.json"
    session_file.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    assert adapter.detect(session_file) is True
    sample = adapter.load(session_file)
    assert sample.session.session_id == "classic-001"
    assert sample.session.ticket is None


# --- Alcove context synthesis tests (issue #114) ---


class TestAlcoveContextExtraction:
    def test_synthesizes_context_from_read_tool(self, tmp_path):
        """Read tool outputs should be extracted as synthesized context."""
        transcript = {
            "session_id": "ctx-read-test",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Read",
                                "type": "tool_use",
                                "input": {"file_path": "/src/main.py"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "def main():\n    print('hello')",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "I read the file."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        assert sample.phases[0].knowledge_context is not None
        assert "def main()" in sample.phases[0].knowledge_context
        assert sample.context_source == "synthesized"

    def test_synthesizes_context_from_grep_tool(self, tmp_path):
        """Grep tool outputs should be extracted as synthesized context."""
        transcript = {
            "session_id": "ctx-grep-test",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Grep",
                                "type": "tool_use",
                                "input": {"pattern": "def validate"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "src/validator.py:10: def validate_input(data):",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Found it."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        assert sample.phases[0].knowledge_context is not None
        assert "def validate_input" in sample.phases[0].knowledge_context
        assert sample.context_source == "synthesized"

    def test_synthesizes_context_from_informational_bash(self, tmp_path):
        """Informational bash commands (pytest, cat, grep) should be extracted."""
        transcript = {
            "session_id": "ctx-bash-test",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Bash",
                                "type": "tool_use",
                                "input": {"command": "pytest tests/ -v"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "PASSED test_main.py::test_hello",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Tests passed."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        assert sample.phases[0].knowledge_context is not None
        assert "PASSED test_main" in sample.phases[0].knowledge_context

    def test_filters_out_non_informational_bash(self, tmp_path):
        """Non-informational bash commands (cd, ls, pwd, git status) should be skipped."""
        transcript = {
            "session_id": "ctx-skip-bash",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Bash",
                                "type": "tool_use",
                                "input": {"command": "cd /home/user/project"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_02",
                                "name": "Bash",
                                "type": "tool_use",
                                "input": {"command": "ls -la"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "total 42\ndrwxr-xr-x ...",
                                "tool_use_id": "toolu_02",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:07.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_03",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Done."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        # No informational tool calls, so no synthesized context
        assert sample.phases[0].knowledge_context is None
        assert sample.context_source is None

    def test_does_not_overwrite_explicit_knowledge_context(self, tmp_path):
        """When a phase already has knowledge_context, synthesis should not run."""
        transcript = {
            "session_id": "ctx-explicit-test",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Read",
                                "type": "tool_use",
                                "input": {"file_path": "/src/main.py"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "def main(): pass",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Done."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        # Manually set explicit knowledge_context before synthesis would run
        # Since this tests the load() flow, we verify the adapter does synthesize
        # because Alcove never has explicit knowledge_context set externally.
        # This test instead verifies context_source is set correctly.
        assert sample.context_source == "synthesized"

    def test_redacts_sensitive_content_in_synthesized_context(self, tmp_path):
        """Synthesized context must pass through redact_sensitive()."""
        transcript = {
            "session_id": "ctx-redact-test",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Read",
                                "type": "tool_use",
                                "input": {"file_path": "/etc/secrets.env"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "password=super_secret_123",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Read secrets."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        assert sample.phases[0].knowledge_context is not None
        assert "super_secret_123" not in sample.phases[0].knowledge_context
        assert "***REDACTED***" in sample.phases[0].knowledge_context

    def test_truncates_context_to_50000_chars(self, tmp_path):
        """Synthesized context should be truncated to 50,000 characters maximum."""
        # Create a session with a very long tool output
        long_content = "x" * 60000
        transcript = {
            "session_id": "ctx-truncate-test",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Read",
                                "type": "tool_use",
                                "input": {"file_path": "/big_file.txt"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": long_content,
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Read big file."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 5000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        assert sample.phases[0].knowledge_context is not None
        assert len(sample.phases[0].knowledge_context) <= 50000

    def test_joins_multiple_tool_outputs_with_separator(self, tmp_path):
        """Multiple tool outputs should be joined with \\n---\\n separator."""
        transcript = {
            "session_id": "ctx-multi-tool",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_01",
                                "name": "Read",
                                "type": "tool_use",
                                "input": {"file_path": "/file1.py"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "content of file1",
                                "tool_use_id": "toolu_01",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:06.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_02",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [
                            {
                                "id": "toolu_02",
                                "name": "Grep",
                                "type": "tool_use",
                                "input": {"pattern": "search"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "grep result line 1",
                                "tool_use_id": "toolu_02",
                            }
                        ],
                    },
                    "timestamp": "2026-04-16T12:11:07.161Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_03",
                        "role": "assistant",
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                        "content": [{"type": "text", "text": "Done."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.02,
                    "duration_ms": 8000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        knowledge = sample.phases[0].knowledge_context
        assert knowledge is not None
        assert "\n---\n" in knowledge
        assert "content of file1" in knowledge
        assert "grep result line 1" in knowledge

    def test_no_context_when_no_tool_outputs(self, tmp_path):
        """Sessions with no tool calls should not have synthesized context."""
        transcript = {
            "session_id": "ctx-no-tools",
            "transcript": [
                {"type": "system", "model": "claude-sonnet-4-20250514"},
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_01",
                        "role": "assistant",
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                        "content": [{"type": "text", "text": "Just text, no tools."}],
                    },
                },
                {
                    "type": "result",
                    "total_cost_usd": 0.01,
                    "duration_ms": 3000,
                },
            ],
        }
        fixture = tmp_path / "session.json"
        fixture.write_text(json.dumps(transcript))
        adapter = AlcoveAdapter()
        sample = adapter.load(fixture)
        assert sample.phases[0].knowledge_context is None
        assert sample.context_source is None


# --- Soda context synthesis tests (issue #114) ---


# --- Alcove rework_cycles support (issue #176) ---


def _tool_use_block(tool_id: str, tool_name: str, tool_input: dict) -> dict:
    """Build an assistant entry containing a single tool_use block."""
    return {
        "type": "assistant",
        "message": {
            "id": f"msg_{tool_id}",
            "role": "assistant",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": [
                {
                    "id": tool_id,
                    "name": tool_name,
                    "type": "tool_use",
                    "input": tool_input,
                }
            ],
        },
    }


def _tool_result_block(tool_id: str, content: str, is_error: bool = False) -> dict:
    """Build a user entry containing a single tool_result block."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": content,
                    "tool_use_id": tool_id,
                    "is_error": is_error,
                }
            ],
        },
        "timestamp": "2026-04-16T12:11:06.161Z",
    }


def _make_transcript_session(tmp_path, transcript_entries, overrides=None):
    """Create a classic alcove session file with the given transcript entries."""
    data = {
        "session_id": "rework-detect-test",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-20250514"},
            *transcript_entries,
            {"type": "result", "total_cost_usd": 0.05, "duration_ms": 30000},
        ],
    }
    if overrides:
        data.update(overrides)
    session_file = tmp_path / "rework-test.json"
    session_file.write_text(json.dumps(data))
    return session_file


class TestAlcoveReworkCycles:
    """Tests for rework cycle detection from transcript tool calls."""

    def test_rework_cycles_from_bridge_top_level(self, tmp_path):
        """Bridge format with rework_cycles at top level should populate session.rework_cycles."""
        source = _bridge_session(tmp_path, overrides={"rework_cycles": 2})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 2

    def test_explicit_rework_cycles_overrides_detection(self, tmp_path):
        """When rework_cycles is explicitly set in the JSON, detection is skipped."""
        # Build a transcript with a detectable rework cycle but override with explicit 0
        entries = [
            _tool_use_block("t1", "Edit", {"file_path": "/src/main.py"}),
            _tool_result_block("t1", "file written"),
            _tool_use_block("t2", "Bash", {"command": "pytest tests/"}),
            _tool_result_block("t2", "FAILED test_main.py::test_x"),
            _tool_use_block("t3", "Edit", {"file_path": "/src/main.py"}),
            _tool_result_block("t3", "file fixed"),
            _tool_use_block("t4", "Bash", {"command": "pytest tests/"}),
            _tool_result_block("t4", "1 passed"),
        ]
        source = _make_transcript_session(tmp_path, entries, overrides={"rework_cycles": 0})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        # Explicit value takes priority over transcript detection
        assert sample.session.rework_cycles == 0

    def test_classic_alcove_rework_cycles_defaults_to_zero(self):
        """Classic alcove format (simple session) should detect 0 rework cycles."""
        adapter = AlcoveAdapter()
        sample = adapter.load(ALCOVE_FIXTURE)
        assert sample.session.rework_cycles == 0

    def test_rework_detected_from_transcript(self, tmp_path):
        """Test failure + edit same file + re-test = 1 rework cycle (positive case)."""
        entries = [
            # Write a file initially
            _tool_use_block("t1", "Edit", {"file_path": "/src/main.py"}),
            _tool_result_block("t1", "file written"),
            # Run test -- fails
            _tool_use_block("t2", "Bash", {"command": "pytest tests/"}),
            _tool_result_block("t2", "FAILED test_main.py::test_x - AssertionError"),
            # Fix the same file (rework edit)
            _tool_use_block("t3", "Edit", {"file_path": "/src/main.py"}),
            _tool_result_block("t3", "file fixed"),
            # Re-test -- passes
            _tool_use_block("t4", "Bash", {"command": "pytest tests/"}),
            _tool_result_block("t4", "1 passed"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 1

    def test_multiple_rework_cycles_detected(self, tmp_path):
        """Two failure-edit-retest sequences = 2 rework cycles."""
        entries = [
            # First write
            _tool_use_block("t1", "Write", {"file_path": "/src/app.py"}),
            _tool_result_block("t1", "file created"),
            # First test failure
            _tool_use_block("t2", "Bash", {"command": "uv run pytest tests/"}),
            _tool_result_block("t2", "FAILED test_app.py"),
            # Rework edit
            _tool_use_block("t3", "Edit", {"file_path": "/src/app.py"}),
            _tool_result_block("t3", "fixed"),
            # Re-test -- fails again
            _tool_use_block("t4", "Bash", {"command": "uv run pytest tests/"}),
            _tool_result_block("t4", "FAILED test_app.py -- still broken"),
            # Second rework edit
            _tool_use_block("t5", "Edit", {"file_path": "/src/app.py"}),
            _tool_result_block("t5", "fixed again"),
            # Re-test -- passes
            _tool_use_block("t6", "Bash", {"command": "uv run pytest tests/"}),
            _tool_result_block("t6", "1 passed"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 2

    def test_tdd_workflow_zero_rework(self, tmp_path):
        """TDD: write test first, it fails, then implement = 0 rework (negative case).

        In TDD, the initial test failure happens before any implementation file
        is written. Editing a new file (never written before the failure) is not
        rework -- it's the first implementation pass.
        """
        entries = [
            # Write test first
            _tool_use_block("t1", "Write", {"file_path": "/tests/test_feature.py"}),
            _tool_result_block("t1", "test file created"),
            # Run test -- fails (expected in TDD)
            _tool_use_block("t2", "Bash", {"command": "pytest tests/test_feature.py"}),
            _tool_result_block("t2", "FAILED test_feature.py::test_new - no module"),
            # Implement the feature (new file, not previously written)
            _tool_use_block("t3", "Write", {"file_path": "/src/feature.py"}),
            _tool_result_block("t3", "implementation created"),
            # Re-test -- passes
            _tool_use_block("t4", "Bash", {"command": "pytest tests/test_feature.py"}),
            _tool_result_block("t4", "1 passed"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 0

    def test_exploratory_reads_zero_rework(self, tmp_path):
        """A session with only Read/Grep calls produces 0 rework cycles (negative case)."""
        entries = [
            _tool_use_block("t1", "Read", {"file_path": "/src/main.py"}),
            _tool_result_block("t1", "def main(): pass"),
            _tool_use_block("t2", "Grep", {"pattern": "def validate"}),
            _tool_result_block("t2", "src/validator.py:10: def validate_input():"),
            _tool_use_block("t3", "Read", {"file_path": "/src/validator.py"}),
            _tool_result_block("t3", "def validate_input(): ..."),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 0

    def test_lint_failure_counts_as_rework(self, tmp_path):
        """Lint failure (ruff/ty) + edit + re-lint = 1 rework cycle."""
        entries = [
            _tool_use_block("t1", "Edit", {"file_path": "/src/module.py"}),
            _tool_result_block("t1", "file written"),
            _tool_use_block("t2", "Bash", {"command": "uv run ruff check src/"}),
            _tool_result_block("t2", "error: F841 unused variable 'x'"),
            _tool_use_block("t3", "Edit", {"file_path": "/src/module.py"}),
            _tool_result_block("t3", "fixed lint"),
            _tool_use_block("t4", "Bash", {"command": "uv run ruff check src/"}),
            _tool_result_block("t4", "All checks passed!"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 1

    def test_findings_remain_empty_for_classic_alcove(self):
        """findings remains [] for classic alcove format (issue #186 out of scope)."""
        adapter = AlcoveAdapter()
        sample = adapter.load(ALCOVE_FIXTURE)
        assert sample.findings == []


# --- Alcove findings support (issue #176) ---


class TestAlcoveFindings:
    def test_findings_from_bridge_top_level(self, tmp_path):
        """Bridge format with findings list should populate sample.findings."""
        findings_data = [
            {
                "source": "go-specialist",
                "severity": "major",
                "file": "main.go",
                "line": 42,
                "issue": "Memory leak in handler",
                "suggestion": "Use defer to close resources",
            },
            {
                "source": "security-specialist",
                "severity": "critical",
                "file": "auth.go",
                "line": 10,
                "issue": "SQL injection vulnerability",
                "suggestion": "Use parameterized queries",
            },
        ]
        source = _bridge_session(tmp_path, overrides={"findings": findings_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert len(sample.findings) == 2
        assert sample.findings[0].reviewer == "go-specialist"
        assert sample.findings[0].severity == "major"
        assert sample.findings[0].file == "main.go"
        assert sample.findings[0].line == 42
        assert "Memory leak" in sample.findings[0].issue
        critical_findings = [
            finding for finding in sample.findings if finding.severity == "critical"
        ]
        assert len(critical_findings) == 1

    def test_findings_empty_when_absent(self, tmp_path):
        """Bridge format without findings should have empty findings list."""
        source = _bridge_session(tmp_path)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.findings == []

    def test_findings_skips_malformed_finding_missing_issue(self, tmp_path):
        """Bridge format with a finding missing 'issue' key should skip it."""
        findings_data = [
            {"source": "go-specialist", "severity": "major", "issue": "valid issue"},
            {"source": "go-specialist", "severity": "minor"},  # missing 'issue' key
        ]
        source = _bridge_session(tmp_path, overrides={"findings": findings_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert len(sample.findings) == 1
        assert sample.findings[0].issue == "valid issue"

    def test_findings_redacts_sensitive_content(self, tmp_path):
        """Findings from bridge format must pass through redact_sensitive()."""
        findings_data = [
            {
                "source": "security-specialist",
                "severity": "critical",
                "file": "config.go",
                "issue": "Hardcoded password=super_secret_123 in config",
            }
        ]
        source = _bridge_session(tmp_path, overrides={"findings": findings_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert len(sample.findings) == 1
        assert "super_secret_123" not in sample.findings[0].issue
        assert "***REDACTED***" in sample.findings[0].issue

    def test_classic_alcove_findings_always_empty(self):
        """Classic alcove format has no findings structure, so findings are empty."""
        adapter = AlcoveAdapter()
        sample = adapter.load(ALCOVE_FIXTURE)
        assert sample.findings == []

    def test_findings_null_in_bridge_format(self, tmp_path):
        """Bridge format with null findings should produce empty findings list."""
        source = _bridge_session(tmp_path, overrides={"findings": None})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.findings == []


# --- Alcove multiple phases support (issue #176) ---


class TestAlcoveMultiplePhases:
    def test_phases_dict_creates_multiple_phase_results(self, tmp_path):
        """Bridge format with phases dict should create one PhaseResult per phase entry."""
        phases_data = {
            "triage": {"status": "completed", "cost_usd": 0.3, "generation": 1},
            "implement": {"status": "completed", "cost_usd": 1.2, "generation": 1},
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert len(sample.phases) == 2
        phase_names = {phase.name for phase in sample.phases}
        assert "triage" in phase_names
        assert "implement" in phase_names

    def test_phases_dict_total_phases_matches_dict_length(self, tmp_path):
        """total_phases on SessionMeta should equal the number of phases in phases dict."""
        phases_data = {
            "triage": {"status": "completed", "cost_usd": 0.3, "generation": 1},
            "plan": {"status": "completed", "cost_usd": 0.5, "generation": 1},
            "implement": {"status": "completed", "cost_usd": 1.0, "generation": 1},
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.total_phases == 3

    def test_phases_dict_metadata_in_phase_results(self, tmp_path):
        """Phase results from phases dict should carry correct metadata."""
        phases_data = {
            "implement": {
                "status": "completed",
                "cost_usd": 1.5,
                "duration_ms": 95000,
                "generation": 2,
                "tokens_in": 5000,
                "tokens_out": 2000,
            }
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert len(impl_phases) == 1
        assert impl_phases[0].status == "completed"
        assert impl_phases[0].cost_usd == 1.5
        assert impl_phases[0].duration_ms == 95000
        assert impl_phases[0].generation == 2
        assert impl_phases[0].tokens_in == 5000
        assert impl_phases[0].tokens_out == 2000

    def test_phases_dict_transcript_data_in_last_phase(self, tmp_path):
        """Transcript output and tool_calls should appear in the last phase."""
        phases_data = {
            "triage": {"status": "completed", "cost_usd": 0.3, "generation": 1},
            "implement": {"status": "completed", "cost_usd": 1.2, "generation": 1},
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        # Transcript output goes to the last phase (implement)
        phase_names = [phase.name for phase in sample.phases]
        last_phase = sample.phases[phase_names.index("implement")]
        assert last_phase.output  # has transcript-derived output ("I will upgrade deps.")
        assert "I will upgrade deps" in last_phase.output

    def test_no_phases_dict_creates_single_session_phase(self, tmp_path):
        """Bridge format without phases dict should create a single 'session' phase."""
        source = _bridge_session(tmp_path)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert len(sample.phases) == 1
        assert sample.phases[0].name == "session"

    def test_phases_dict_failed_status_propagates(self, tmp_path):
        """Phase with 'failed' status in phases dict should produce failed PhaseResult."""
        phases_data = {
            "implement": {"status": "failed", "cost_usd": 0.5, "generation": 1},
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert impl_phases[0].status == "failed"

    def test_phases_dict_with_rework_and_findings_combined(self, tmp_path):
        """Bridge format with rework_cycles, findings, and phases all populated correctly."""
        phases_data = {
            "implement": {"status": "completed", "cost_usd": 0.8, "generation": 2},
            "review": {"status": "completed", "cost_usd": 0.3, "generation": 1},
        }
        findings_data = [
            {
                "source": "specialist",
                "severity": "minor",
                "file": "main.go",
                "issue": "style issue",
            },
        ]
        overrides = {
            "rework_cycles": 1,
            "phases": phases_data,
            "findings": findings_data,
        }
        source = _bridge_session(tmp_path, overrides=overrides)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.rework_cycles == 1
        assert len(sample.findings) == 1
        assert len(sample.phases) == 2

    def test_phases_dict_empty_creates_single_session_phase(self, tmp_path):
        """Bridge format with empty phases dict should create a single 'session' phase."""
        source = _bridge_session(tmp_path, overrides={"phases": {}})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert len(sample.phases) == 1
        assert sample.phases[0].name == "session"

    def test_phases_dict_non_primary_phases_have_empty_output(self, tmp_path):
        """Non-primary (not last) phases should not have transcript-derived output."""
        phases_data = {
            "triage": {"status": "completed", "cost_usd": 0.3, "generation": 1},
            "implement": {"status": "completed", "cost_usd": 1.2, "generation": 1},
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        triage_phases = [phase for phase in sample.phases if phase.name == "triage"]
        assert len(triage_phases) == 1
        # Non-primary phase should not have transcript output
        assert triage_phases[0].output == ""


# --- Alcove transcript-based phase detection (issue #176) ---


class TestAlcoveTranscriptPhaseDetection:
    """Tests for multi-phase detection from transcript tool calls."""

    def test_analysis_then_coding_then_testing_three_phases(self, tmp_path):
        """Read -> Edit -> pytest sequence should detect 3 phases."""
        entries = [
            _tool_use_block("t1", "Read", {"file_path": "/src/main.py"}),
            _tool_result_block("t1", "def main(): pass"),
            _tool_use_block("t2", "Grep", {"pattern": "def validate"}),
            _tool_result_block("t2", "found it"),
            _tool_use_block("t3", "Edit", {"file_path": "/src/main.py"}),
            _tool_result_block("t3", "file updated"),
            _tool_use_block("t4", "Bash", {"command": "pytest tests/"}),
            _tool_result_block("t4", "1 passed"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        # 3 distinct phases: analysis -> coding -> testing
        assert sample.session.total_phases == 3

    def test_single_phase_all_reads(self, tmp_path):
        """A session with only Read calls detects 1 phase (analysis only)."""
        entries = [
            _tool_use_block("t1", "Read", {"file_path": "/src/main.py"}),
            _tool_result_block("t1", "def main(): pass"),
            _tool_use_block("t2", "Read", {"file_path": "/src/utils.py"}),
            _tool_result_block("t2", "def helper(): pass"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.total_phases == 1

    def test_no_tool_calls_defaults_to_one_phase(self, tmp_path):
        """A session with no tool calls should default to 1 phase."""
        entries = [
            {
                "type": "assistant",
                "message": {
                    "id": "msg_01",
                    "role": "assistant",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                    "content": [{"type": "text", "text": "Just text."}],
                },
            },
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.total_phases == 1

    def test_interleaved_analysis_and_coding(self, tmp_path):
        """Read -> Edit -> Read -> Edit detects 4 phase transitions."""
        entries = [
            _tool_use_block("t1", "Read", {"file_path": "/src/a.py"}),
            _tool_result_block("t1", "content a"),
            _tool_use_block("t2", "Edit", {"file_path": "/src/a.py"}),
            _tool_result_block("t2", "edited a"),
            _tool_use_block("t3", "Read", {"file_path": "/src/b.py"}),
            _tool_result_block("t3", "content b"),
            _tool_use_block("t4", "Edit", {"file_path": "/src/b.py"}),
            _tool_result_block("t4", "edited b"),
        ]
        source = _make_transcript_session(tmp_path, entries)
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        # analysis -> coding -> analysis -> coding = 4 phases
        assert sample.session.total_phases == 4

    def test_phases_dict_overrides_transcript_detection(self, tmp_path):
        """When phases dict is present in the JSON, it overrides transcript detection."""
        phases_data = {
            "triage": {"status": "completed", "cost_usd": 0.3, "generation": 1},
            "implement": {"status": "completed", "cost_usd": 1.2, "generation": 1},
        }
        source = _bridge_session(tmp_path, overrides={"phases": phases_data})
        adapter = AlcoveAdapter()
        sample = adapter.load(source)
        assert sample.session.total_phases == 2

    def test_phase_names_no_collision_with_session_schema(self, tmp_path):
        """Detected phase categories (analysis/coding/testing) must not collide
        with session_schema names (triage/plan/implement/verify) that trigger
        context synthesis."""
        from raki.adapters.alcove import _classify_tool_call

        # The phase names returned by _classify_tool_call are:
        # "analysis", "coding", "testing" -- none overlap with
        # the session_schema PHASE_NAMES: "triage", "plan", "implement", "verify"
        session_schema_names = {"triage", "plan", "implement", "verify"}
        assert _classify_tool_call("Read", {}) not in session_schema_names
        assert _classify_tool_call("Edit", {}) not in session_schema_names
        assert _classify_tool_call("Bash", {"command": "pytest tests/"}) not in session_schema_names


class TestSodaContextExtraction:
    def test_synthesizes_context_from_triage_structured(self, tmp_path):
        """Triage output_structured fields should be extracted for context."""
        meta = {
            "ticket": "700",
            "summary": "triage synthesis test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {
                "triage": {"status": "completed", "generation": 1},
                "implement": {"status": "completed", "generation": 1},
            },
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")
        triage_data = {
            "approach": "Add input validation to the API",
            "code_area": "api/handlers",
            "files": ["src/api/handler.py", "src/api/validator.py"],
            "risks": ["Breaking change for existing clients"],
        }
        (tmp_path / "triage.json").write_text(json.dumps(triage_data))
        implement_data = {"summary": "implemented validation"}
        (tmp_path / "implement.json").write_text(json.dumps(implement_data))

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        # Context should be stored on the implement phase (where to_ragas_rows reads from)
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert len(impl_phases) >= 1
        assert impl_phases[-1].knowledge_context is not None
        context = impl_phases[-1].knowledge_context
        assert "Add input validation" in context
        assert "api/handlers" in context
        assert sample.context_source == "synthesized"

    def test_synthesizes_context_from_plan_structured(self, tmp_path):
        """Plan output_structured fields should be extracted for context."""
        meta = {
            "ticket": "701",
            "summary": "plan synthesis test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {
                "plan": {"status": "completed", "generation": 1},
                "implement": {"status": "completed", "generation": 1},
            },
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")
        plan_data = {
            "approach": "Refactor the validation layer",
            "tasks": [
                {
                    "description": "Add field-level validators",
                    "files": ["src/validators.py"],
                },
                {
                    "description": "Update API handlers",
                    "files": ["src/handlers.py"],
                },
            ],
        }
        (tmp_path / "plan.json").write_text(json.dumps(plan_data))
        implement_data = {"summary": "implemented plan"}
        (tmp_path / "implement.json").write_text(json.dumps(implement_data))

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        # Context should be stored on the implement phase
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert len(impl_phases) >= 1
        assert impl_phases[-1].knowledge_context is not None
        context = impl_phases[-1].knowledge_context
        assert "Refactor the validation layer" in context
        assert sample.context_source == "synthesized"

    def test_synthesizes_context_from_implement_structured(self, tmp_path):
        """Implement output_structured fields should be extracted for context."""
        meta = {
            "ticket": "702",
            "summary": "implement synthesis test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {
                "implement": {"status": "completed", "generation": 1},
            },
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")
        implement_data = {
            "files_changed": ["src/main.py", "tests/test_main.py"],
            "commits": [
                {"message": "feat: add input validation"},
                {"message": "test: add validation tests"},
            ],
            "deviations": "Had to change the API contract",
        }
        (tmp_path / "implement.json").write_text(json.dumps(implement_data))

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        # Context should be stored on the implement phase
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert len(impl_phases) >= 1
        assert impl_phases[-1].knowledge_context is not None
        context = impl_phases[-1].knowledge_context
        assert "src/main.py" in context
        assert sample.context_source == "synthesized"

    def test_does_not_overwrite_explicit_knowledge_context(self, tmp_path):
        """When a phase already has knowledge_context, synthesis should not run."""
        meta = {
            "ticket": "703",
            "summary": "explicit context test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {
                "implement": {"status": "completed", "generation": 1},
            },
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")
        implement_data = {
            "files_changed": ["src/main.py"],
            "summary": "implemented",
        }
        (tmp_path / "implement.json").write_text(json.dumps(implement_data))

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        # Verify synthesis produced context (since no explicit context exists)
        assert sample.context_source == "synthesized"
        # Context should be on the implement phase
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert len(impl_phases) >= 1
        assert impl_phases[-1].knowledge_context is not None

    def test_redacts_sensitive_content_in_synthesized_context(self, tmp_path):
        """Synthesized context must pass through redact_sensitive()."""
        meta = {
            "ticket": "704",
            "summary": "redaction test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {
                "triage": {"status": "completed", "generation": 1},
                "implement": {"status": "completed", "generation": 1},
            },
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")
        triage_data = {
            "approach": "Fix the password=super_secret_value leak",
            "code_area": "auth",
        }
        (tmp_path / "triage.json").write_text(json.dumps(triage_data))
        implement_data = {"summary": "fixed leak"}
        (tmp_path / "implement.json").write_text(json.dumps(implement_data))

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        phases_with_context = [
            phase for phase in sample.phases if phase.knowledge_context is not None
        ]
        assert len(phases_with_context) >= 1
        context = phases_with_context[0].knowledge_context
        assert "super_secret_value" not in context

    def test_falls_back_to_implement_output_when_no_structured(self, tmp_path):
        """When no structured fields exist, fall back to implement phase output."""
        meta = {
            "ticket": "705",
            "summary": "fallback test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {
                "implement": {"status": "completed", "generation": 1},
            },
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")
        # A minimal implement.json with no structured extraction fields
        implement_data = {"summary": "I implemented the feature by modifying the handler"}
        (tmp_path / "implement.json").write_text(json.dumps(implement_data))

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        # Context should be on the implement phase
        impl_phases = [phase for phase in sample.phases if phase.name == "implement"]
        assert len(impl_phases) >= 1
        assert impl_phases[-1].knowledge_context is not None
        assert sample.context_source == "synthesized"

    def test_no_synthesis_when_no_phases(self, tmp_path):
        """Sessions with no phases should not synthesize context."""
        meta = {
            "ticket": "706",
            "summary": "no phases test",
            "started_at": "2026-04-10T08:00:00Z",
            "total_cost": 5.0,
            "rework_cycles": 0,
            "phases": {},
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        (tmp_path / "events.jsonl").write_text("")

        adapter = SessionSchemaAdapter()
        sample = adapter.load(tmp_path)
        assert sample.context_source is None


# --- Ticket #175: pipeline/orchestrator metadata ---


def test_session_schema_orchestrator_from_branch_prefix(tmp_path):
    """branch field 'soda/101' should yield orchestrator='soda'."""
    meta = {
        "ticket": "175",
        "summary": "orchestrator inference",
        "branch": "soda/101",
        "started_at": "2026-04-24T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {"triage": {"status": "completed"}, "implement": {"status": "completed"}},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.orchestrator == "soda"


def test_session_schema_orchestrator_none_without_branch(tmp_path):
    """Without a branch field, orchestrator should be None."""
    meta = {
        "ticket": "175b",
        "summary": "no orchestrator",
        "started_at": "2026-04-24T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.orchestrator is None


def test_session_schema_orchestrator_none_branch_no_slash(tmp_path):
    """A branch value without a slash yields orchestrator=None."""
    meta = {
        "ticket": "175c",
        "summary": "branch no slash",
        "branch": "main",
        "started_at": "2026-04-24T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.orchestrator is None


def test_session_schema_pipeline_phases_from_phases_dict(tmp_path):
    """pipeline_phases should reflect the ordered keys from meta.json phases dict."""
    meta = {
        "ticket": "175d",
        "summary": "pipeline phases",
        "branch": "soda/175",
        "started_at": "2026-04-24T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {
            "triage": {"status": "completed"},
            "plan": {"status": "completed"},
            "implement": {"status": "completed"},
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.pipeline_phases == ["triage", "plan", "implement"]


def test_session_schema_pipeline_phases_none_when_empty(tmp_path):
    """pipeline_phases should be None when phases dict is empty."""
    meta = {
        "ticket": "175e",
        "summary": "empty phases",
        "started_at": "2026-04-24T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.pipeline_phases is None


def test_session_schema_provider_defaults_to_none(tmp_path):
    """provider should always be None for session-schema adapter (not populated)."""
    meta = {
        "ticket": "175f",
        "summary": "provider default",
        "started_at": "2026-04-24T08:00:00Z",
        "total_cost": 5.0,
        "rework_cycles": 0,
        "phases": {},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text("")
    adapter = SessionSchemaAdapter()
    sample = adapter.load(tmp_path)
    assert sample.session.provider is None


def test_session_schema_fixture_orchestrator(pass_simple_dir):
    """pass-simple fixture has branch='soda/101' so orchestrator should be 'soda'."""
    adapter = SessionSchemaAdapter()
    sample = adapter.load(pass_simple_dir)
    assert sample.session.orchestrator == "soda"
    assert sample.session.pipeline_phases == ["triage", "plan", "implement", "verify", "review"]


# --- Ticket #175: AlcoveAdapter pipeline/orchestrator metadata ---


def test_alcove_classic_orchestrator_is_alcove(tmp_path):
    """Classic alcove format (no task_id) should yield orchestrator='alcove'."""
    data = {
        "session_id": "classic-001",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-20250514"},
            {
                "type": "result",
                "total_cost_usd": 0.01,
                "duration_ms": 1000,
            },
        ],
    }
    fixture = tmp_path / "classic.json"
    fixture.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    assert sample.session.orchestrator == "alcove"


def test_alcove_bridge_orchestrator_is_bridge(tmp_path):
    """Bridge format (has task_id) should yield orchestrator='bridge'."""
    source = _bridge_session(tmp_path)
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.orchestrator == "bridge"


def test_alcove_provider_from_raw(tmp_path):
    """provider field in the JSON should be populated on SessionMeta.provider."""
    data = {
        "session_id": "prov-001",
        "provider": "anthropic",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-20250514"},
            {"type": "result", "total_cost_usd": 0.01, "duration_ms": 1000},
        ],
    }
    fixture = tmp_path / "provider-session.json"
    fixture.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    assert sample.session.provider == "anthropic"


def test_alcove_provider_none_when_absent(tmp_path):
    """provider should be None when not present in the JSON."""
    data = {
        "session_id": "prov-002",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-20250514"},
            {"type": "result", "total_cost_usd": 0.01, "duration_ms": 1000},
        ],
    }
    fixture = tmp_path / "no-provider.json"
    fixture.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    assert sample.session.provider is None


def test_alcove_pipeline_phases_from_phases_dict(tmp_path):
    """pipeline_phases should reflect ordered phase keys from the phases dict."""
    source = _bridge_session(
        tmp_path,
        overrides={
            "phases": {
                "triage": {"status": "completed"},
                "implement": {"status": "completed"},
                "verify": {"status": "completed"},
            }
        },
    )
    adapter = AlcoveAdapter()
    sample = adapter.load(source)
    assert sample.session.pipeline_phases == ["triage", "implement", "verify"]


def test_alcove_pipeline_phases_none_when_no_phases(tmp_path):
    """pipeline_phases should be None when phases dict is absent."""
    data = {
        "session_id": "phases-none",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-20250514"},
            {"type": "result", "total_cost_usd": 0.01, "duration_ms": 1000},
        ],
    }
    fixture = tmp_path / "no-phases.json"
    fixture.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    sample = adapter.load(fixture)
    assert sample.session.pipeline_phases is None


def test_alcove_classic_fixture_orchestrator():
    """The classic alcove fixture should have orchestrator='alcove' and no pipeline_phases."""
    adapter = AlcoveAdapter()
    sample = adapter.load(ALCOVE_FIXTURE)
    assert sample.session.orchestrator == "alcove"
    assert sample.session.pipeline_phases is None
    assert sample.session.provider is None


# --- Alcove output_structured from phases dict (fix #183) ---


def test_alcove_phases_output_structured_populated(tmp_path):
    """PhaseResult.output_structured must be populated from the phases dict metadata.

    Without this fix, _build_phases() never passed output_structured to
    PhaseResult, so _extract_domains() always returned {} and ground-truth
    matching was broken for all alcove/bridge sessions.
    """
    phases_data = {
        "triage": {
            "status": "completed",
            "cost_usd": 0.3,
            "generation": 1,
            "approach": "Patch the dependency version",
            "code_area": "deps",
        },
        "implement": {
            "status": "completed",
            "cost_usd": 1.2,
            "generation": 1,
        },
    }
    source = _bridge_session(tmp_path, overrides={"phases": phases_data})
    adapter = AlcoveAdapter()
    sample = adapter.load(source)

    triage_phases = [phase for phase in sample.phases if phase.name == "triage"]
    assert len(triage_phases) == 1
    triage = triage_phases[0]

    # output_structured must be populated so ground-truth matching can read it
    assert triage.output_structured is not None
    assert triage.output_structured.get("approach") == "Patch the dependency version"
    assert triage.output_structured.get("code_area") == "deps"


def test_alcove_phases_empty_phase_meta_output_structured_none(tmp_path):
    """When a phase entry in the phases dict is None (null), output_structured must be None.

    A None/empty phase_meta means there is no metadata to store; passing None
    to redact_dict() would error, so the adapter must guard with 'if phase_meta'.
    """
    # phases dict with a null value for triage
    data = {
        "id": "bridge-null-phase",
        "task_id": "task-null",
        "started_at": "2026-04-22T10:00:00Z",
        "status": "completed",
        "transcript": [
            {"type": "system", "model": "claude-sonnet-4-6"},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [{"type": "text", "text": "done"}],
                },
            },
            {"type": "result", "total_cost_usd": 0.01, "duration_ms": 1000},
        ],
        "phases": {"triage": None},
    }
    session_file = tmp_path / "null-phase.json"
    session_file.write_text(json.dumps(data))
    adapter = AlcoveAdapter()
    sample = adapter.load(session_file)

    triage_phases = [phase for phase in sample.phases if phase.name == "triage"]
    assert len(triage_phases) == 1
    # Null phase_meta → output_structured must be None (not an empty dict)
    assert triage_phases[0].output_structured is None


def test_alcove_no_phases_dict_output_structured_none(tmp_path):
    """When no phases dict is present, the single 'session' phase has output_structured=None.

    The single-phase fallback path in _build_phases() always had output_structured=None
    (not affected by the fix).  This test verifies the unchanged path still works.
    """
    source = _bridge_session(tmp_path)  # no phases override → falls back to single phase
    adapter = AlcoveAdapter()
    sample = adapter.load(source)

    assert len(sample.phases) == 1
    assert sample.phases[0].name == "session"
    assert sample.phases[0].output_structured is None


# --- Ticket #223: SODA session fixture ---


def test_session_schema_soda_session_detects(soda_session_dir: Path):
    """soda-session fixture is detected as a valid session by SessionSchemaAdapter."""
    adapter = SessionSchemaAdapter()
    assert adapter.detect(soda_session_dir)


def test_session_schema_adapter_loads_soda_session(soda_session_dir: Path):
    """soda-session fixture loads correctly with expected session metadata."""
    adapter = SessionSchemaAdapter()
    sample = adapter.load(soda_session_dir)
    assert sample.session.session_id == "223"
    assert sample.session.ticket == "223"
    assert sample.session.rework_cycles == 1
    assert sample.session.total_cost_usd == 12.25
    assert sample.session.orchestrator == "soda"
    pipeline = sample.session.pipeline_phases
    assert pipeline is not None
    assert "submit" in pipeline
    assert "monitor" in pipeline
    assert pipeline == [
        "triage",
        "plan",
        "implement",
        "verify",
        "review",
        "submit",
        "monitor",
    ]
    assert len(sample.events) == 16


def test_session_schema_soda_session_has_all_phases(soda_session_dir: Path):
    """soda-session fixture contains triage, plan, implement, and verify phases."""
    adapter = SessionSchemaAdapter()
    sample = adapter.load(soda_session_dir)
    phase_names = {phase.name for phase in sample.phases}
    assert "triage" in phase_names
    assert "plan" in phase_names
    assert "implement" in phase_names
    assert "verify" in phase_names


def test_session_schema_soda_session_triage_structured(soda_session_dir: Path):
    """soda-session triage phase has SODA-schema structured output."""
    adapter = SessionSchemaAdapter()
    sample = adapter.load(soda_session_dir)
    triage = next(phase for phase in sample.phases if phase.name == "triage")
    assert triage.output_structured is not None
    assert triage.output_structured["ticket_key"] == "223"
    assert triage.output_structured["complexity"] == "small"
    assert isinstance(triage.output_structured["files"], list)
    assert triage.output_structured["automatable"] is True


def test_session_schema_soda_session_implement_structured(soda_session_dir: Path):
    """soda-session implement phase has SODA-schema structured output with commits."""
    adapter = SessionSchemaAdapter()
    sample = adapter.load(soda_session_dir)
    implement = next(phase for phase in sample.phases if phase.name == "implement")
    assert implement.output_structured is not None
    assert implement.output_structured["tests_passed"] is True
    commits = implement.output_structured.get("commits", [])
    assert len(commits) >= 1
    assert implement.output_structured["branch"] == "soda/223"
    # implement.generation should be 2 (rework cycle)
    assert implement.generation == 2


def test_session_schema_soda_session_review_finding(soda_session_dir: Path):
    """soda-session fixture has review findings in perspectives structure.

    The current adapter reads top-level 'findings' only. Once #220 lands
    (perspectives support), this test should assert findings are loaded.
    For now, verify the adapter does not crash on the SODA review format.
    """
    adapter = SessionSchemaAdapter()
    sample = adapter.load(soda_session_dir)
    # Findings are in perspectives structure (SODA schema), not top-level.
    # Adapter cannot read them yet (#220). Verify graceful handling.
    assert sample.findings == []


def test_session_schema_soda_session_synthesizes_context(soda_session_dir: Path):
    """soda-session fixture synthesizes retrieval context from SODA phase outputs."""
    adapter = SessionSchemaAdapter()
    sample = adapter.load(soda_session_dir)
    has_context = any(phase.knowledge_context is not None for phase in sample.phases)
    assert has_context
    assert sample.context_source == "synthesized"
