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
    description: str = "Mean USD cost per session"
    rationale: str = (
        "LLM API cost is the most direct financial measure of agent efficiency. Unlike the "
        "other metrics, cost does not have a universal target threshold because acceptable "
        "cost depends on the value of the work being automated. Instead, it is tracked to "
        "identify outliers and trends: sessions that cost significantly more than the median "
        "are candidates for investigation. High cost combined with high rework_cycles confirms "
        "that iteration is the primary cost driver; high cost with low rework_cycles points to "
        "large context windows or verbose tool usage as the culprit. The metric reports raw USD "
        "rather than a normalized score because cost thresholds are project-specific. "
        "N/A is shown when no session has cost data (total_cost_usd not logged)."
    )

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
