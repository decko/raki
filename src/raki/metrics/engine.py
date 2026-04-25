import uuid
from collections.abc import Sequence

from raki.metrics.health import run_health_checks
from raki.metrics.protocol import Metric, MetricConfig, TokenAccumulator
from raki.model import EvalDataset
from raki.model.report import EvalReport, MetricResult, MetricWarning, SampleResult


class MetricsEngine:
    def __init__(self, metrics: Sequence[Metric], config: MetricConfig | None = None) -> None:
        self._metrics = metrics
        self._config = config or MetricConfig()

    def run(
        self,
        dataset: EvalDataset,
        skip_judge: bool = False,
        skip_ground_truth: bool = False,
    ) -> EvalReport:
        accumulator = TokenAccumulator()
        self._config.token_accumulator = accumulator

        results: list[MetricResult] = []
        for metric in self._metrics:
            if skip_judge and metric.requires_llm:
                continue
            if skip_ground_truth and metric.requires_ground_truth:
                continue
            result = metric.compute(dataset, self._config)
            results.append(result)
        aggregate = {result.name: result.score for result in results}
        details = {result.name: result.details for result in results if result.details}
        sample_results = self._build_sample_results(dataset, results)
        llm_used = not skip_judge

        # Run health checks for all computed metrics and collect warnings.
        total_sessions = len(dataset.samples)
        all_warnings: list[MetricWarning] = []
        for result in results:
            all_warnings.extend(run_health_checks(result, total_sessions))

        report_config: dict = {
            "llm_provider": self._config.llm_provider if llm_used else None,
            "llm_model": self._config.llm_model if llm_used else None,
            "llm_temperature": self._config.temperature if llm_used else None,
            "llm_max_tokens": self._config.max_tokens if llm_used else None,
            "metrics": [metric.name for metric in self._metrics],
            "skip_judge": skip_judge,
        }

        if accumulator.calls > 0:
            report_config["judge_cost"] = {
                "input_tokens": accumulator.input_tokens,
                "output_tokens": accumulator.output_tokens,
                "calls": accumulator.calls,
            }

        return EvalReport(
            run_id=f"eval-{uuid.uuid4().hex[:8]}",
            config=report_config,
            aggregate_scores=aggregate,
            metric_details=details,
            sample_results=sample_results,
            manifest_hash=dataset.manifest_hash,
            warnings=all_warnings,
        )

    @staticmethod
    def _build_sample_results(
        dataset: EvalDataset,
        metric_results: list[MetricResult],
    ) -> list[SampleResult]:
        """Build one SampleResult per session from per-metric sample_scores.

        Only metrics that populate ``sample_scores`` with a per-session key are
        included in each :class:`SampleResult`.  Aggregate-only metrics (e.g.
        ``ReviewSeverityDistribution``) that
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
