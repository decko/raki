"""Triage calibration metric.

Measures how well the agent's triage complexity prediction aligns with the actual
session cost. A session is "calibrated" when the predicted complexity level is
consistent with the actual total_cost_usd.

Thresholds:
- small  → actual cost must be <= SMALL_MAX (8.0 USD)
- medium → actual cost must be <= MEDIUM_MAX (16.0 USD)
- large  → any cost is acceptable (no upper bound)

Score = mean(calibrated sessions) over sessions with both a triage complexity
prediction and a cost.  Returns score=None (N/A) when no sessions qualify.

This is a purely operational metric -- no LLM required.
"""

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult

# Cost thresholds (USD) for complexity calibration.
SMALL_MAX: float = 8.0
MEDIUM_MAX: float = 16.0


class TriageCalibrationMetric:
    """Fraction of sessions where triage complexity prediction matches actual cost.

    Score = calibrated_sessions / sessions_with_triage_and_cost.
    Higher is better: 1.0 means every prediction was calibrated,
    0.0 means every prediction was wrong.
    Returns score=None when no sessions have both a triage complexity and cost.
    """

    name: str = "triage_calibration"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = True
    display_format: str = "percent"
    display_name: str = "Triage calibration"
    description: str = "Fraction of sessions where predicted complexity matches actual cost"
    rationale: str = (
        "Triage calibration measures whether the agent's upfront complexity estimate "
        "predicts the actual cost of completing the session. A well-calibrated agent "
        "labels small tasks cheaply and large tasks expensively, enabling accurate "
        "planning and resource allocation. Miscalibration in either direction is a "
        "signal: small-labeled sessions that cost a lot suggest the agent underestimates "
        "effort; large-labeled sessions that cost very little suggest over-caution. "
        "Only sessions with both a triage complexity estimate and an actual cost are "
        "scored; sessions missing either are excluded (N/A). "
        "Thresholds: small <= $8.00, medium <= $16.00, large: any cost. "
        "Target: >= 80% calibrated."
    )

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:  # noqa: ARG002
        small_max = SMALL_MAX
        medium_max = MEDIUM_MAX

        calibrated = 0
        total = 0
        sample_scores: dict[str, float] = {}
        per_complexity: dict[str, dict[str, int]] = {
            "small": {"total": 0, "calibrated": 0},
            "medium": {"total": 0, "calibrated": 0},
            "large": {"total": 0, "calibrated": 0},
        }

        for sample in dataset.samples:
            # Find the triage phase (take the first one found).
            triage_phase = None
            for phase in sample.phases:
                if phase.name == "triage":
                    triage_phase = phase
                    break

            if triage_phase is None:
                continue

            output = triage_phase.output_structured
            if not isinstance(output, dict):
                continue

            complexity = output.get("complexity")
            if complexity not in ("small", "medium", "large"):
                continue

            cost = sample.session.total_cost_usd
            if cost is None:
                continue

            total += 1
            per_complexity[complexity]["total"] += 1

            if complexity == "small":
                is_calibrated = cost <= small_max
            elif complexity == "medium":
                is_calibrated = cost <= medium_max
            else:
                is_calibrated = True

            session_score = 1.0 if is_calibrated else 0.0
            sample_scores[sample.session.session_id] = session_score
            if is_calibrated:
                calibrated += 1
                per_complexity[complexity]["calibrated"] += 1

        if total == 0:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "calibrated_sessions": 0,
                    "sessions_with_triage_and_cost": 0,
                    "small_max": small_max,
                    "medium_max": medium_max,
                    "per_complexity": per_complexity,
                },
            )

        score = calibrated / total
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "calibrated_sessions": calibrated,
                "sessions_with_triage_and_cost": total,
                "small_max": small_max,
                "medium_max": medium_max,
                "per_complexity": per_complexity,
            },
            sample_scores=sample_scores,
        )
