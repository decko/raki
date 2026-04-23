"""Tests for docs/metrics/rationale-and-interpretation.md.

Verifies that the rationale and interpretation guide exists and contains
the expected sections for all non-Ragas metrics.
"""

from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs" / "metrics"
RATIONALE_DOC = DOCS_DIR / "rationale-and-interpretation.md"

NON_RAGAS_METRICS = [
    "first_pass_verify_rate",
    "rework_cycles",
    "review_severity_distribution",
    "cost_efficiency",
    "self_correction_rate",
    "phase_execution_time",
    "token_efficiency",
    "knowledge_gap_rate",
    "knowledge_miss_rate",
]


class TestRationaleDocExists:
    def test_doc_file_exists(self):
        assert RATIONALE_DOC.exists(), "docs/metrics/rationale-and-interpretation.md does not exist"

    def test_doc_is_non_empty(self):
        assert RATIONALE_DOC.stat().st_size > 500, "Doc is unexpectedly short"


class TestRationaleDocContent:
    @pytest.fixture(autouse=True)
    def load_doc(self):
        self.content = RATIONALE_DOC.read_text(encoding="utf-8")

    def test_has_title(self):
        assert "# " in self.content, "Doc must have at least one heading"

    @pytest.mark.parametrize("metric_name", NON_RAGAS_METRICS)
    def test_metric_name_present(self, metric_name):
        assert metric_name in self.content, (
            f"Metric '{metric_name}' not mentioned in rationale-and-interpretation.md"
        )

    def test_has_rationale_section_for_each_metric(self):
        """Each metric should have a section explaining the rationale."""
        for metric_name in NON_RAGAS_METRICS:
            assert metric_name in self.content, f"'{metric_name}' not found in rationale doc"

    def test_doc_links_to_operational_reference(self):
        assert "operational.md" in self.content, (
            "Doc should cross-reference the operational metrics reference"
        )

    def test_doc_links_to_knowledge_reference(self):
        assert "knowledge.md" in self.content, (
            "Doc should cross-reference the knowledge metrics reference"
        )

    def test_doc_has_interpretation_guidance(self):
        """Doc must include interpretation-related keywords."""
        interpretation_terms = ["interpret", "signal", "action", "target"]
        found = [term for term in interpretation_terms if term.lower() in self.content.lower()]
        assert len(found) >= 3, (
            f"Doc lacks interpretation guidance. Missing terms from: {interpretation_terms}"
        )

    def test_doc_mentions_zone_colors(self):
        """Doc should reference the green/yellow/red zone system."""
        assert any(color in self.content.lower() for color in ["green", "yellow", "red"]), (
            "Doc should explain the green/yellow/red zone thresholds"
        )

    def test_doc_has_combined_pattern_section(self):
        """Doc should include combined metric pattern analysis."""
        assert any(
            phrase in self.content.lower()
            for phrase in ["pattern", "combination", "combined", "together", "both"]
        ), "Doc should include guidance on reading metrics in combination"
