from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class ReviewSeverityDistribution:
    name: str = "review_severity_distribution"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = True
    display_format: str = "score"
    display_name: str = "Severity score"
    description: str = "Weighted severity of review findings (1.0 = no findings)"
    rationale: str = (
        "Not all findings are equal: a critical finding (broken functionality, security flaw) "
        "is far more damaging than a minor style nit. The weighted severity score assigns "
        "critical findings a weight of 3, major a weight of 2, and minor a weight of 1, "
        "then normalizes so that 1.0 represents a completely clean run. This weighting "
        "correctly identifies runs where a handful of critical findings indicate a systemic "
        "problem, even if the total finding count is low. The score is inverted (higher is "
        "better) to follow the convention that improving scores move upward. "
        "A run with 3 critical findings scores lower than one with 30 minor findings, "
        "reflecting real-world impact: critical issues block delivery while minor nits do not. "
        "Target: >=0.85 (few or no significant findings)."
    )

    WEIGHTS: dict[str, int] = {"critical": 3, "major": 2, "minor": 1}

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        counts: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
        for sample in dataset.samples:
            for finding in sample.findings:
                if finding.severity in counts:
                    counts[finding.severity] += 1
        total = sum(counts.values())
        if total > 0:
            weighted = sum(self.WEIGHTS[sev] * count for sev, count in counts.items())
            max_weighted = self.WEIGHTS["critical"] * total
            score = 1.0 - (weighted / max_weighted)
        else:
            score = 1.0
        return MetricResult(
            name=self.name,
            score=score,
            details={**counts, "total": total},
        )
