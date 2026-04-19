"""Knowledge retrieval miss rate metric.

For each rework-triggering finding (critical or major severity), classifies
whether the issue stems from a retrieval gap (relevant knowledge was not
retrieved) or a capability gap (knowledge was present but not applied).

This is a purely operational metric -- no LLM required.
"""

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset, EvalSample
from raki.model.report import MetricResult


class KnowledgeRetrievalMissRate:
    """Checks whether rework-triggering findings could have been prevented
    by knowledge that was (or wasn't) in knowledge_context.

    Score = retrieval_gaps / total_rework_findings.
    Lower is better: 0.0 means all gaps are capability (knowledge was available),
    1.0 means all gaps are retrieval (knowledge was missing).
    """

    name: str = "knowledge_retrieval_miss_rate"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "score"
    display_name: str = "Knowledge miss rate"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        retrieval_gaps = 0
        capability_gaps = 0
        total_rework_findings = 0

        for sample in dataset.samples:
            if sample.session.rework_cycles == 0:
                continue

            knowledge_text = self._extract_knowledge_context(sample)

            for finding in sample.findings:
                if finding.severity not in ("critical", "major"):
                    continue
                total_rework_findings += 1

                # Word-boundary matching via set intersection avoids partial
                # substring matches (e.g. "auth" matching "author").  Words must
                # be longer than 4 characters to reduce false positives from
                # common short words like "does", "code", "type".
                issue_words_set = {word for word in finding.issue.lower().split() if len(word) > 4}
                knowledge_words = set(knowledge_text.split()) if knowledge_text else set()
                if issue_words_set & knowledge_words:
                    capability_gaps += 1  # knowledge present but not applied
                else:
                    retrieval_gaps += 1  # no related knowledge retrieved

        score = retrieval_gaps / total_rework_findings if total_rework_findings > 0 else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "retrieval_gaps": retrieval_gaps,
                "capability_gaps": capability_gaps,
                "total_rework_findings": total_rework_findings,
            },
        )

    @staticmethod
    def _extract_knowledge_context(sample: EvalSample) -> str | None:
        """Extract knowledge_context text from the latest implement or session phase."""
        matching = [phase for phase in sample.phases if phase.name in ("implement", "session")]
        if not matching:
            return None
        phase = max(matching, key=lambda phase: phase.generation)
        if phase.knowledge_context is None:
            return None
        return phase.knowledge_context.lower()
