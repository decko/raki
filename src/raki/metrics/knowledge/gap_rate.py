"""Knowledge gap rate metric.

Measures the ratio of rework-triggering findings (critical/major) in domains
NOT covered by the knowledge base.

When doc chunks are provided (via ``--docs-path``), domain matching is
per-domain: a finding is "uncovered" only when its issue words do NOT overlap
with the words from any domain's doc chunks.

When no doc chunks are available, falls back to the legacy
``knowledge_context`` text on session phases.

Score = uncovered_findings / total_rework_findings.
Lower is better: 0.0 means all findings are in covered domains,
1.0 means all findings are in domains missing from the KB.

Returns score=None when no rework findings exist, no doc chunks AND no
knowledge_context are available (N/A).

This metric does NOT require an LLM.
"""

from raki.metrics.knowledge._common import (
    build_domain_word_sets,
    extract_knowledge_context,
    is_finding_covered_by_chunks,
    _MIN_WORD_LENGTH,
)
from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class KnowledgeGapRate:
    """Ratio of review findings in domains NOT covered by the knowledge base.

    Score = uncovered_findings / total_rework_findings.
    Lower is better.
    Returns score=None when denominator is zero or no knowledge context exists.
    """

    name: str = "knowledge_gap_rate"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "score"
    display_name: str = "Knowledge gap rate"
    description: str = "Ratio of rework findings in domains not covered by the knowledge base"
    rationale: str = (
        "When an agent fails on a task, the first diagnostic question is: did it have the right "
        "reference material? Knowledge gap rate answers this by checking whether the domains "
        "where critical and major failures occurred are covered by the knowledge base. "
        "A finding is 'uncovered' when its issue words do not overlap with any domain's doc "
        "content. A high gap rate is directly actionable: the uncovered findings point to "
        "specific topics that need to be added to the knowledge base. Unlike Ragas context "
        "precision/recall (which measure retrieval quality within the KB), gap rate measures "
        "coverage: whether the content exists at all. Returns N/A when no rework findings "
        "exist or when no docs are loaded — both are expected for clean, no-KB runs. "
        "Target: <0.20 (KB covers >80% of failure domains)."
    )

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        # When doc chunks are available, use domain-aware matching
        if config.doc_chunks:
            return self._compute_with_doc_chunks(dataset, config)
        return self._compute_with_knowledge_context(dataset)

    def _compute_with_doc_chunks(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        """Compute using per-domain doc chunk matching."""
        domain_word_sets = build_domain_word_sets(config.doc_chunks)

        uncovered_findings = 0
        total_rework_findings = 0

        for sample in dataset.samples:
            if sample.session.rework_cycles == 0:
                continue

            for finding in sample.findings:
                if finding.severity not in ("critical", "major"):
                    continue
                total_rework_findings += 1

                if not is_finding_covered_by_chunks(finding.issue, domain_word_sets):
                    uncovered_findings += 1

        if total_rework_findings == 0:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "uncovered_findings": 0,
                    "total_rework_findings": 0,
                },
            )

        score = uncovered_findings / total_rework_findings
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "uncovered_findings": uncovered_findings,
                "total_rework_findings": total_rework_findings,
            },
        )

    def _compute_with_knowledge_context(self, dataset: EvalDataset) -> MetricResult:
        """Legacy path: compute using phase knowledge_context strings."""
        uncovered_findings = 0
        total_rework_findings = 0
        has_any_knowledge_context = False

        for sample in dataset.samples:
            if sample.session.rework_cycles == 0:
                continue

            knowledge_text = extract_knowledge_context(sample)
            if not knowledge_text:
                continue  # Skip this entire session -- don't count its findings

            has_any_knowledge_context = True

            for finding in sample.findings:
                if finding.severity not in ("critical", "major"):
                    continue
                total_rework_findings += 1

                issue_words = {
                    word for word in finding.issue.lower().split() if len(word) >= _MIN_WORD_LENGTH
                }
                knowledge_words = set(knowledge_text.split())
                if not (issue_words & knowledge_words):
                    uncovered_findings += 1

        if total_rework_findings == 0 or not has_any_knowledge_context:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "uncovered_findings": 0,
                    "total_rework_findings": total_rework_findings,
                },
            )

        score = uncovered_findings / total_rework_findings
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "uncovered_findings": uncovered_findings,
                "total_rework_findings": total_rework_findings,
            },
        )
