from datetime import datetime, timezone
from pathlib import Path

from raki.ground_truth.matcher import load_ground_truth, match_ground_truth
from raki.model import EvalSample, PhaseResult, SessionMeta
from raki.model.ground_truth import GroundTruth

FIXTURES = Path(__file__).parent / "fixtures" / "ground_truth"


def test_load_ground_truth():
    entries = load_ground_truth(FIXTURES / "curated.yaml")
    assert len(entries) == 2
    assert entries[0].question is not None
    assert len(entries[0].domains) > 0


def test_load_ground_truth_preserves_fields():
    entries = load_ground_truth(FIXTURES / "curated.yaml")
    first = entries[0]
    assert first.difficulty == "medium"
    assert first.knowledge_type == "constraint"
    assert first.expected_phase == "implement"
    assert first.expected_contexts is not None
    assert len(first.expected_contexts) == 2


def test_load_ground_truth_filters_unknown_keys():
    """Non-model keys like 'id' and 'source' should be filtered out, not cause errors."""
    entries = load_ground_truth(FIXTURES / "curated.yaml")
    assert len(entries) >= 2


def test_load_ground_truth_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    entries = load_ground_truth(empty_file)
    assert entries == []


def test_load_ground_truth_non_list(tmp_path: Path):
    non_list_file = tmp_path / "scalar.yaml"
    non_list_file.write_text("not_a_list: true")
    entries = load_ground_truth(non_list_file)
    assert entries == []


def test_load_ground_truth_skips_non_dict_items(tmp_path: Path):
    """Non-dict items (strings, numbers, nulls) in the YAML list are silently skipped."""
    mixed_file = tmp_path / "mixed.yaml"
    mixed_file.write_text(
        "- just a bare string\n"
        "- 42\n"
        "- null\n"
        "- question: 'Valid entry'\n"
        "  reference_answer: 'An answer'\n"
        "  domains: [testing]\n"
        "- question: 'Another valid entry'\n"
        "  reference_answer: 'Another answer'\n"
        "  domains: [quality]\n"
    )
    entries = load_ground_truth(mixed_file)
    assert len(entries) == 2
    assert entries[0].question == "Valid entry"
    assert entries[1].question == "Another valid entry"


def test_match_ground_truth_by_domain():
    entries = [
        GroundTruth(
            question="How to handle git ops?",
            reference_answer="Use guardrails",
            domains=["git", "guardrails"],
        ),
        GroundTruth(
            question="How to deploy?",
            reference_answer="Use CI",
            domains=["deployment", "ci"],
        ),
    ]
    meta = SessionMeta(
        session_id="1",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )
    triage = PhaseResult(
        name="triage",
        generation=1,
        status="completed",
        output="git operations",
        output_structured={"code_area": "git commands, guardrails"},
    )
    sample = EvalSample(session=meta, phases=[triage], findings=[], events=[])
    matched = match_ground_truth(sample, entries)
    assert matched is not None
    assert "git" in matched.domains


def test_match_ground_truth_no_match():
    entries = [
        GroundTruth(
            question="How to deploy?",
            reference_answer="Use CI",
            domains=["deployment"],
        ),
    ]
    meta = SessionMeta(
        session_id="1",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )
    triage = PhaseResult(
        name="triage",
        generation=1,
        status="completed",
        output="database migration",
        output_structured={"code_area": "database"},
    )
    sample = EvalSample(session=meta, phases=[triage], findings=[], events=[])
    matched = match_ground_truth(sample, entries)
    assert matched is None


def test_match_ground_truth_picks_best_overlap():
    entries = [
        GroundTruth(
            question="Single domain match",
            reference_answer="Partial",
            domains=["git"],
        ),
        GroundTruth(
            question="Double domain match",
            reference_answer="Better",
            domains=["git", "guardrails"],
        ),
    ]
    meta = SessionMeta(
        session_id="1",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )
    triage = PhaseResult(
        name="triage",
        generation=1,
        status="completed",
        output="git guardrails",
        output_structured={"code_area": "git, guardrails"},
    )
    sample = EvalSample(session=meta, phases=[triage], findings=[], events=[])
    matched = match_ground_truth(sample, entries)
    assert matched is not None
    assert matched.question == "Double domain match"


def test_match_ground_truth_no_triage_phase():
    """If there is no triage phase, no domains are extracted and no match is returned."""
    entries = [
        GroundTruth(
            question="How to deploy?",
            reference_answer="Use CI",
            domains=["deployment"],
        ),
    ]
    meta = SessionMeta(
        session_id="1",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )
    implement = PhaseResult(
        name="implement",
        generation=1,
        status="completed",
        output="done",
    )
    sample = EvalSample(session=meta, phases=[implement], findings=[], events=[])
    matched = match_ground_truth(sample, entries)
    assert matched is None


def test_match_ground_truth_no_structured_output():
    """If triage has no output_structured, no domains are extracted."""
    entries = [
        GroundTruth(
            question="How to deploy?",
            reference_answer="Use CI",
            domains=["deployment"],
        ),
    ]
    meta = SessionMeta(
        session_id="1",
        started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        total_phases=1,
        rework_cycles=0,
    )
    triage = PhaseResult(
        name="triage",
        generation=1,
        status="completed",
        output="deployment stuff",
    )
    sample = EvalSample(session=meta, phases=[triage], findings=[], events=[])
    matched = match_ground_truth(sample, entries)
    assert matched is None
