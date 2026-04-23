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
    description: str = "Mean review-rework iterations per session"
    rationale: str = (
        "Rework cycles measure the cost of iteration: each cycle represents a review-fix loop "
        "where the agent consumed additional tokens, introduced latency, and potentially added "
        "new defects. A session with rework_cycles=3 is roughly 3x more expensive in LLM calls "
        "and wall-clock time than a first-pass success. This metric uses the raw count rather "
        "than a normalized 0-1 score because the business impact scales linearly with cycles. "
        "Unlike first_pass_verify_rate (which is binary per session), rework_cycles captures "
        "the degree of iteration: an agent averaging 0.2 cycles is significantly more efficient "
        "than one averaging 1.2 cycles, even if both occasionally fail the first pass. "
        "Target: <1.5 cycles on average."
    )

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
