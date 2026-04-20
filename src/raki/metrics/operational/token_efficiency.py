import statistics

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class TokenEfficiencyMetric:
    name: str = "token_efficiency"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = False
    display_format: str = "count"
    display_name: str = "Tokens / phase"
    description: str = "Average tokens (in + out) per phase"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        sample_scores: dict[str, float] = {}
        session_averages: list[float] = []

        for sample in dataset.samples:
            phase_totals: list[int] = []
            for phase in sample.phases:
                if phase.tokens_in is None and phase.tokens_out is None:
                    continue
                phase_totals.append((phase.tokens_in or 0) + (phase.tokens_out or 0))

            if not phase_totals:
                continue

            avg_per_phase = statistics.mean(phase_totals)
            session_averages.append(avg_per_phase)
            sample_scores[sample.session.session_id] = avg_per_phase

        mean_avg = statistics.mean(session_averages) if session_averages else 0.0

        details: dict[str, float | int] = {
            "mean_tokens_per_phase": mean_avg,
            "sessions_with_tokens": len(session_averages),
        }

        return MetricResult(
            name=self.name,
            score=mean_avg,
            details=details,
            sample_scores=sample_scores,
        )
