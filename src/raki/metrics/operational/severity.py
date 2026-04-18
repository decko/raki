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
