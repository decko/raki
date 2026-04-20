import uuid
from collections.abc import Sequence

from raki.metrics.protocol import Metric, MetricConfig
from raki.model import EvalDataset
from raki.model.report import EvalReport, MetricResult, SampleResult


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
        details = {result.name: result.details for result in results if result.details}
        sample_results = self._build_sample_results(dataset, results)
        return EvalReport(
            run_id=f"eval-{uuid.uuid4().hex[:8]}",
            config={
                "llm_model": self._config.llm_model,
                "metrics": [metric.name for metric in self._metrics],
                "skip_llm": skip_llm,
            },
            aggregate_scores=aggregate,
            metric_details=details,
            sample_results=sample_results,
            manifest_hash=dataset.manifest_hash,
        )

    @staticmethod
    def _build_sample_results(
        dataset: EvalDataset,
        metric_results: list[MetricResult],
    ) -> list[SampleResult]:
        """Build one SampleResult per session from per-metric sample_scores.

        Only metrics that populate ``sample_scores`` with a per-session key are
        included in each :class:`SampleResult`.  Aggregate-only metrics (e.g.
        ``ReviewSeverityDistribution``, ``KnowledgeRetrievalMissRate``) that
        leave ``sample_scores`` empty will appear in
        :attr:`EvalReport.aggregate_scores` but **not** in the per-session
        drill-down returned here.
        """
        sample_results: list[SampleResult] = []
        for sample in dataset.samples:
            session_id = sample.session.session_id
            per_metric_scores: list[MetricResult] = []
            for metric_result in metric_results:
                if session_id in metric_result.sample_scores:
                    per_metric_scores.append(
                        MetricResult(
                            name=metric_result.name,
                            score=metric_result.sample_scores[session_id],
                        )
                    )
            sample_results.append(
                SampleResult(
                    sample=sample,
                    scores=per_metric_scores,
                )
            )
        return sample_results

    def run_single(self, metric_name: str, dataset: EvalDataset) -> MetricResult:
        for metric in self._metrics:
            if metric.name == metric_name:
                return metric.compute(dataset, self._config)
        raise ValueError(f"Unknown metric: {metric_name}")
