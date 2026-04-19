"""Tests for the example ground truth entries in examples/ground-truth/curated.yaml."""

from pathlib import Path

import pytest

from raki.ground_truth.matcher import load_ground_truth

EXAMPLE_GROUND_TRUTH = Path(__file__).parent.parent / "examples" / "ground-truth" / "curated.yaml"


@pytest.fixture(scope="module")
def ground_truth_entries():
    return load_ground_truth(EXAMPLE_GROUND_TRUTH)


def test_example_ground_truth_loads_five_entries(ground_truth_entries):
    assert len(ground_truth_entries) == 5


def test_example_ground_truth_covers_all_difficulty_levels(ground_truth_entries):
    difficulties = {entry.difficulty for entry in ground_truth_entries}
    assert difficulties == {"easy", "medium", "hard"}


def test_example_ground_truth_covers_all_knowledge_types(ground_truth_entries):
    knowledge_types = {entry.knowledge_type for entry in ground_truth_entries}
    assert knowledge_types == {"fact", "procedure", "constraint", "context-dependent"}


def test_example_ground_truth_has_nonempty_domains(ground_truth_entries):
    for entry in ground_truth_entries:
        assert len(entry.domains) > 0, f"Entry with question {entry.question!r} has empty domains"


def test_example_ground_truth_has_nonempty_question_and_reference_answer(ground_truth_entries):
    for entry in ground_truth_entries:
        assert entry.question, f"Entry has empty or missing question: {entry!r}"
        assert entry.reference_answer, (
            f"Entry with question {entry.question!r} has empty or missing reference_answer"
        )


def test_example_ground_truth_has_nonempty_expected_contexts(ground_truth_entries):
    for entry in ground_truth_entries:
        assert entry.expected_contexts, (
            f"Entry with question {entry.question!r} has empty or missing expected_contexts"
        )
