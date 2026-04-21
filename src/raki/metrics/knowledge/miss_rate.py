"""Knowledge miss rate metric (redefined).

Measures the ratio of rework-triggering findings (critical/major) in domains
that ARE covered by the knowledge base but the agent still got wrong.
A finding is "covered" when its issue words overlap with the knowledge_context text.

Score = covered_findings / total_rework_findings.
Lower is better: 0.0 means no findings in covered domains (the KB is working),
1.0 means all findings are in covered domains (the agent ignored the KB).

Returns score=None when no rework findings exist or no knowledge_context
is available (N/A).

This metric does NOT require an LLM.
"""

from raki.metrics.knowledge._common import extract_knowledge_context
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

                issue_words = {word for word in finding.issue.lower().split() if len(word) > 4}
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
