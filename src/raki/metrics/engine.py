import uuid
from collections.abc import Sequence

from raki.metrics.protocol import Metric, MetricConfig
from raki.model import EvalDataset
from raki.model.report import EvalReport, MetricResult


class MetricsEngine:
    def __init__(self, metrics: Sequence[Metric], config: MetricConfig | None = None) -> None:
        self._metrics = metrics
        self._config = config or MetricConfig()

    def run(
        self,
        dataset: EvalDataset,
        skip_llm: bool = False,
        skip_ground_truth: bool = False,
    ) -> EvalReport:
        results: list[MetricResult] = []
        for metric in self._metrics:
            if skip_llm and metric.requires_llm:
                continue
            if skip_ground_truth and metric.requires_ground_truth:
                continue
            result = metric.compute(dataset, self._config)
            results.append(result)
        aggregate = {result.name: result.score for result in results}
        return EvalReport(
            run_id=f"eval-{uuid.uuid4().hex[:8]}",
            config={
                "llm_model": self._config.llm_model,
                "metrics": [metric.name for metric in self._metrics],
                "skip_llm": skip_llm,
            },
            aggregate_scores=aggregate,
            sample_results=[],
            manifest_hash=dataset.manifest_hash,
        )

    def run_single(self, metric_name: str, dataset: EvalDataset) -> MetricResult:
        for metric in self._metrics:
            if metric.name == metric_name:
                return metric.compute(dataset, self._config)
        raise ValueError(f"Unknown metric: {metric_name}")
