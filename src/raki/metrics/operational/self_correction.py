"""Self-correction rate metric.

Measures how effectively an agent resolves rework findings.
Score = resolved_findings / total_rework_findings across all sessions with rework.

A finding is considered "resolved" when the session's final verify phase
has status "completed". If no rework findings exist, returns score=None (N/A).

This is a purely operational metric -- no LLM required.
"""

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset
from raki.model.report import MetricResult


class SelfCorrectionRate:
    """Ratio of rework findings that were resolved by the agent.

    Score = resolved_findings / total_rework_findings.
    Higher is better: 1.0 means all rework findings were resolved,
    0.0 means none were resolved.
    Returns score=None when no rework findings exist (N/A).
    """

    name: str = "self_correction_rate"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = True
    display_format: str = "percent"
    display_name: str = "Self-correction rate"
    description: str = "Ratio of rework findings resolved by the agent"
    rationale: str = (
        "When an agent makes a mistake, the critical question is: can it fix it? "
        "Self-correction rate measures whether the agent can apply reviewer feedback and "
        "deliver a correct result in subsequent generations. A high self-correction rate "
        "means the agent is effectively learning from feedback within a session. A low rate "
        "means the agent is churning—consuming tokens and time without converging on a correct "
        "answer. Only critical and major findings are counted because minor findings rarely "
        "block final verification. A session's findings are considered resolved when its "
        "final verify phase has status='completed'. Returns N/A (not 0.0) when no rework "
        "findings exist, which is the normal state for high-quality runs. "
        "Target: >=80% of rework findings resolved."
    )

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:
        total_rework_findings = 0
        resolved_findings = 0
        sample_scores: dict[str, float] = {}

        for sample in dataset.samples:
            if sample.session.rework_cycles == 0:
                continue

            session_findings = [
                finding
                for finding in sample.findings
                if finding.severity in ("critical", "major")
                # Synthesized findings are not actionable review feedback; exclude
                # them from self-correction rate so repeated test failures don't
                # artificially inflate the denominator.
                and finding.finding_source != "synthesized"
            ]
            session_findings_count = len(session_findings)
            if session_findings_count == 0:
                continue

            total_rework_findings += session_findings_count

            # Check if the final verify phase completed successfully
            verify_phases = [phase for phase in sample.phases if phase.name == "verify"]
            if verify_phases:
                final_verify = max(verify_phases, key=lambda phase: phase.generation)
                session_resolved = final_verify.status == "completed"
            else:
                session_resolved = False

            if session_resolved:
                resolved_findings += session_findings_count
                sample_scores[sample.session.session_id] = 1.0
            else:
                sample_scores[sample.session.session_id] = 0.0

        if total_rework_findings == 0:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "total_rework_findings": 0,
                    "resolved_findings": 0,
                },
            )

        score = resolved_findings / total_rework_findings
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "total_rework_findings": total_rework_findings,
                "resolved_findings": resolved_findings,
            },
            sample_scores=sample_scores,
        )
