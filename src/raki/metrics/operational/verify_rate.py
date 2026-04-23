from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class FirstPassVerifyRate:
    name: str = "first_pass_verify_rate"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = True
    display_format: str = "percent"
    display_name: str = "Verify rate"
    description: str = "% sessions passing verify on first try"
    rationale: str = (
        "The first-pass verify rate is the primary signal of implementation quality. "
        "An agent that consistently delivers correct work on the first attempt is more reliable "
        "and less expensive than one requiring multiple review cycles. This metric focuses on "
        "generation=1 because subsequent verify phases represent rework, which is already "
        "captured by rework_cycles. A session is counted as a first-pass success only when "
        "the generation-1 verify phase has status='completed'; a failed generation-1 verify "
        "scores 0.0 for that session even if a later generation eventually passes. "
        "Target: >85% first-pass success."
    )

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        sample_scores: dict[str, float] = {}
        passed = 0
        total = 0
        for sample in dataset.samples:
            verify_phases = [phase for phase in sample.phases if phase.name == "verify"]
            if not verify_phases:
                continue
            min_gen = min(phase.generation for phase in verify_phases)
            first_pass = min_gen == 1 and any(
                phase.generation == 1 and phase.status == "completed" for phase in verify_phases
            )
            total += 1
            if first_pass:
                passed += 1
                sample_scores[sample.session.session_id] = 1.0
            else:
                sample_scores[sample.session.session_id] = 0.0
        score = passed / total if total > 0 else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            details={"passed": passed, "total": total},
            sample_scores=sample_scores,
        )
