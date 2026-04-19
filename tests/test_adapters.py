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
