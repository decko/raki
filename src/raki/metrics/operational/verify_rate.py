"""First-pass success rate metric.

Measures what fraction of sessions completed without any rework cycles.
Score = sessions_with_zero_rework / total_sessions.

This replaces the old FirstPassVerifyRate which inspected verify phase
generation numbers and could produce contradictory signals vs rework_cycles.
By reading rework_cycles directly from SessionMeta the two metrics are
guaranteed to be consistent.

Returns score=None for empty datasets (no applicable data).
"""

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class FirstPassSuccessRate:
    """Fraction of sessions that completed without any rework cycles.

    Score = sessions_with_rework_cycles_0 / total_sessions.
    Higher is better: 1.0 means every session passed on the first attempt,
    0.0 means every session required at least one rework cycle.
    Returns score=None when the dataset is empty (N/A).
    """

    name: str = "first_pass_success_rate"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = True
    display_format: str = "percent"
    display_name: str = "First-pass success rate"
    description: str = "% sessions with no rework cycles"

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        sample_scores: dict[str, float] = {}
        passed = 0
        total = len(dataset.samples)
        for sample in dataset.samples:
            if sample.session.rework_cycles == 0:
                passed += 1
                sample_scores[sample.session.session_id] = 1.0
            else:
                sample_scores[sample.session.session_id] = 0.0
        if total == 0:
            return MetricResult(
                name=self.name,
                score=None,
                details={"passed": 0, "total": 0},
            )
        score = passed / total
        return MetricResult(
            name=self.name,
            score=score,
            details={"passed": passed, "total": total},
            sample_scores=sample_scores,
        )
