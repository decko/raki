import statistics

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class PhaseExecutionTimeMetric:
    name: str = "phase_execution_time"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "duration"
    display_name: str = "Phase execution time"
    description: str = "Mean total phase execution time per session (seconds)"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        sample_scores: dict[str, float] = {}
        session_totals: list[float] = []

        for sample in dataset.samples:
            durations = [
                phase.duration_ms for phase in sample.phases if phase.duration_ms is not None
            ]
            if not durations:
                continue
            total_seconds = sum(durations) / 1000.0
            session_totals.append(total_seconds)
            sample_scores[sample.session.session_id] = total_seconds

        mean_total = statistics.mean(session_totals) if session_totals else 0.0
        sorted_totals = sorted(session_totals) if session_totals else []

        details: dict[str, float | int] = {
            "mean": mean_total,
            "sessions_with_duration": len(session_totals),
        }
        if sorted_totals:
            details["min"] = sorted_totals[0]
            details["max"] = sorted_totals[-1]
            details["p50"] = statistics.median(sorted_totals)
            idx_95 = min(int(len(sorted_totals) * 0.95), len(sorted_totals) - 1)
            details["p95"] = sorted_totals[idx_95]

        return MetricResult(
            name=self.name,
            score=mean_total,
            details=details,
            sample_scores=sample_scores,
        )
