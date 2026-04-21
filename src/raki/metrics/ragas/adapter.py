"""Adapter from EvalDataset to Ragas-compatible row format.

Maps EvalSample fields to Ragas 0.4 collections API ascore() keyword arg
names without importing ragas -- this module has no ragas dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raki.model import EvalDataset, EvalSample
from raki.model.phases import PhaseResult

if TYPE_CHECKING:
    from raki.docs.chunker import DocChunk

# Maximum characters for each retrieved context chunk sent to Ragas.
# Synthesized contexts from session transcripts can be 100k+ chars;
# truncating prevents max_tokens errors during Ragas scoring.
MAX_CONTEXT_CHARS: int = 10_000

# Maximum characters for the response field sent to Ragas.
# Implement phase output can contain thousands of lines of raw code/JSON;
# Ragas faithfulness decomposes this into individual statements, which explodes.
MAX_RESPONSE_CHARS: int = 10_000

_TRUNCATION_MARKER = " [truncated]"


def truncate_for_ragas(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate text to *max_chars*, cutting at a word boundary.

    When truncation occurs a ``[truncated]`` marker is appended so
    downstream consumers know content was removed.  The marker is *not*
    counted against the limit -- total length may be up to
    ``max_chars + len(" [truncated]")``.
    """
    if len(text) <= max_chars:
        return text

    # Cut at max_chars then find the last space to avoid splitting a word.
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + _TRUNCATION_MARKER


@dataclass
class RagasRow:
    """Internal representation matching Ragas 0.4 collections API ascore() kwargs."""

    session_id: str
    user_input: str  # maps to ascore() kwarg user_input
    retrieved_contexts: list[str]  # maps to ascore() kwarg retrieved_contexts
    response: str  # maps to ascore() kwarg response
    reference: str | None  # maps to ascore() kwarg reference (was ground_truths in v0.3)


def to_ragas_rows(
    dataset: EvalDataset,
    doc_chunks: list[DocChunk] | None = None,
) -> list[RagasRow]:
    """Extract RagasRow objects from an EvalDataset.

    Skips samples that have no implement/session phase or no knowledge_context.

    When *doc_chunks* are provided and a sample has no ground-truth
    ``reference_answer``, the doc-chunk texts are joined and used as the
    ``reference`` field so that precision/recall metrics can compute real
    scores instead of returning N/A.
    """
    doc_reference: str | None = None
    if doc_chunks:
        doc_reference = "\n\n".join(chunk.text for chunk in doc_chunks)

    rows: list[RagasRow] = []
    for sample in dataset.samples:
        implement = _find_phase(sample, "implement") or _find_phase(sample, "session")
        if implement is None or implement.knowledge_context is None:
            continue
        contexts = [
            truncate_for_ragas(chunk.strip(), max_chars=MAX_CONTEXT_CHARS)
            for chunk in implement.knowledge_context.split("\n---\n")
            if chunk.strip()
        ]
        if not contexts:
            continue
        user_input = _extract_question(sample)
        response = truncate_for_ragas(implement.output, max_chars=MAX_RESPONSE_CHARS)
        reference = None
        if sample.ground_truth and sample.ground_truth.reference_answer:
            reference = sample.ground_truth.reference_answer
        elif doc_reference:
            reference = doc_reference
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


def is_max_tokens_error(exc: Exception) -> bool:
    """Check whether *exc* indicates the LLM hit its output token limit.

    Anthropic and OpenAI surface this as ``max_tokens`` in the error
    message or stop reason.  We do a case-insensitive substring check
    so we catch variants across providers.
    """
    message = str(exc).lower()
    return "max_tokens" in message


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
