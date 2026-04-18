from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class ReworkCycles:
    name: str = "rework_cycles"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "count"
    display_name: str = "Rework cycles"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        sample_scores: dict[str, float] = {}
        total_rework = 0
        count = len(dataset.samples)
        for sample in dataset.samples:
            cycles = sample.session.rework_cycles
            sample_scores[sample.session.session_id] = float(cycles)
            total_rework += cycles
        mean = total_rework / count if count > 0 else 0.0
        return MetricResult(
            name=self.name,
            score=mean,
            details={"total_rework": total_rework, "sessions": count},
            sample_scores=sample_scores,
        )
