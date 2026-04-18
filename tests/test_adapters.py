"""Tests for the adapter layer: protocol, registry, loader, session-schema adapter."""

import json
from pathlib import Path

from raki.adapters.loader import DatasetLoader
from raki.adapters.registry import AdapterRegistry
from raki.adapters.session_schema import SessionSchemaAdapter


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
