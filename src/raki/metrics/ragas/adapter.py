"""Adapter from EvalDataset to Ragas-compatible row format.

Maps EvalSample fields to Ragas 0.4 collections API ascore() keyword arg
names without importing ragas -- this module has no ragas dependency.
"""

from dataclasses import dataclass

from raki.model import EvalDataset, EvalSample
from raki.model.phases import PhaseResult


@dataclass
class RagasRow:
    """Internal representation matching Ragas 0.4 collections API ascore() kwargs."""

    session_id: str
    user_input: str  # maps to ascore() kwarg user_input
    retrieved_contexts: list[str]  # maps to ascore() kwarg retrieved_contexts
    response: str  # maps to ascore() kwarg response
    reference: str | None  # maps to ascore() kwarg reference (was ground_truths in v0.3)


def to_ragas_rows(dataset: EvalDataset) -> list[RagasRow]:
    """Extract RagasRow objects from an EvalDataset.

    Skips samples that have no implement/session phase or no knowledge_context.
    """
    rows: list[RagasRow] = []
    for sample in dataset.samples:
        implement = _find_phase(sample, "implement") or _find_phase(sample, "session")
        if implement is None or implement.knowledge_context is None:
            continue
        contexts = [
            chunk.strip() for chunk in implement.knowledge_context.split("\n---\n") if chunk.strip()
        ]
        if not contexts:
            continue
        user_input = _extract_question(sample)
        response = implement.output
        reference = None
        if sample.ground_truth and sample.ground_truth.reference_answer:
            reference = sample.ground_truth.reference_answer
        rows.append(
            RagasRow(
                session_id=sample.session.session_id,
                user_input=user_input,
                retrieved_contexts=contexts,
                response=response,
                reference=reference,
            )
        )
    return rows


def _find_phase(sample: EvalSample, name: str) -> PhaseResult | None:
    """Find the latest generation of a named phase."""
    phases = [phase for phase in sample.phases if phase.name == name]
    if not phases:
        return None
    return max(phases, key=lambda phase: phase.generation)


def _extract_question(sample: EvalSample) -> str:
    """Extract a question/task description from the sample.

    Priority:
    1. Ground truth question
    2. Triage approach or summary from output_structured
    3. Session ticket or session_id as fallback
    """
    if sample.ground_truth and sample.ground_truth.question:
        return sample.ground_truth.question
    for phase in sample.phases:
        if phase.name == "triage" and phase.output_structured:
            approach = phase.output_structured.get("approach", "")
            if approach:
                return approach
            summary = phase.output_structured.get("summary", "")
            if summary:
                return summary
    return sample.session.ticket or sample.session.session_id


def detect_context_source(dataset: EvalDataset) -> str | None:
    """Determine the predominant context_source across dataset samples.

    Returns "synthesized" if any sample used synthesized context,
    "explicit" if all samples used explicit context, or None if no samples
    have context_source set.
    """
    sources = {sample.context_source for sample in dataset.samples if sample.context_source}
    if not sources:
        return None
    if "synthesized" in sources:
        return "synthesized"
    return "explicit"
