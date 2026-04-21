"""Knowledge gap rate metric.

Measures the ratio of rework-triggering findings (critical/major) in domains
NOT covered by the knowledge base. A finding is "uncovered" when its issue
words do not overlap with the knowledge_context text.

Score = uncovered_findings / total_rework_findings.
Lower is better: 0.0 means all findings are in covered domains,
1.0 means all findings are in domains missing from the KB.

Returns score=None when no rework findings exist or no knowledge_context
is available (N/A).

This metric does NOT require an LLM.
"""

from raki.metrics.knowledge._common import extract_knowledge_context
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

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
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

                issue_words = {word for word in finding.issue.lower().split() if len(word) > 4}
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
