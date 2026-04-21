"""Knowledge miss rate metric (redefined).

Measures the ratio of rework-triggering findings (critical/major) in domains
that ARE covered by the knowledge base but the agent still got wrong.

When doc chunks are provided (via ``--docs-path``), domain matching is
per-domain: a finding is "covered" only when its issue words overlap with the
words from a *specific* domain's doc chunks.  This avoids the false-positive
problem where merging all doc text into one blob causes every finding to match.

When no doc chunks are available, falls back to the legacy
``knowledge_context`` text on session phases.

Score = covered_findings / total_rework_findings.
Lower is better: 0.0 means no findings in covered domains (the KB is working),
1.0 means all findings are in covered domains (the agent ignored the KB).

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


class KnowledgeMissRate:
    """Ratio of review findings in domains that ARE covered by the KB
    but the agent still got wrong.

    Score = covered_findings / total_rework_findings.
    Lower is better.
    Returns score=None when denominator is zero or no knowledge context exists.
    """

    name: str = "knowledge_miss_rate"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "score"
    display_name: str = "Knowledge miss rate"
    description: str = "Ratio of rework findings in domains covered by the KB but still wrong"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        # When doc chunks are available, use domain-aware matching
        if config.doc_chunks:
            return self._compute_with_doc_chunks(dataset, config)
        return self._compute_with_knowledge_context(dataset)

    def _compute_with_doc_chunks(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        """Compute using per-domain doc chunk matching."""
        domain_word_sets = build_domain_word_sets(config.doc_chunks)

        covered_findings = 0
        total_rework_findings = 0

        for sample in dataset.samples:
            if sample.session.rework_cycles == 0:
                continue

            for finding in sample.findings:
                if finding.severity not in ("critical", "major"):
                    continue
                total_rework_findings += 1

                if is_finding_covered_by_chunks(finding.issue, domain_word_sets):
                    covered_findings += 1

        if total_rework_findings == 0:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "covered_findings": 0,
                    "total_rework_findings": 0,
                },
            )

        score = covered_findings / total_rework_findings
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "covered_findings": covered_findings,
                "total_rework_findings": total_rework_findings,
            },
        )

    def _compute_with_knowledge_context(self, dataset: EvalDataset) -> MetricResult:
        """Legacy path: compute using phase knowledge_context strings."""
        covered_findings = 0
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
                if issue_words & knowledge_words:
                    covered_findings += 1

        if total_rework_findings == 0 or not has_any_knowledge_context:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "covered_findings": 0,
                    "total_rework_findings": total_rework_findings,
                },
            )

        score = covered_findings / total_rework_findings
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "covered_findings": covered_findings,
                "total_rework_findings": total_rework_findings,
            },
        )
