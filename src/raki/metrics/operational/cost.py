from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class CostEfficiency:
    name: str = "cost_efficiency"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "currency"
    display_name: str = "Cost / session"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        sample_scores: dict[str, float] = {}
        costs: list[float] = []
        for sample in dataset.samples:
            if sample.session.total_cost_usd is not None:
                costs.append(sample.session.total_cost_usd)
                sample_scores[sample.session.session_id] = sample.session.total_cost_usd
        mean = sum(costs) / len(costs) if costs else 0.0
        return MetricResult(
            name=self.name,
            score=mean,
            details={
                "mean_cost": mean,
                "min_cost": min(costs) if costs else 0.0,
                "max_cost": max(costs) if costs else 0.0,
                "sessions_with_cost": len(costs),
            },
            sample_scores=sample_scores,
        )
