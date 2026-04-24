"""Adapter from EvalDataset to Ragas-compatible row format.

Maps EvalSample fields to Ragas 0.4 collections API ascore() keyword arg
names without importing ragas -- this module has no ragas dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from raki.docs.chunker import DocChunk
from raki.model import EvalDataset, EvalSample
from raki.model.phases import PhaseResult

# Maximum characters for each retrieved context chunk sent to Ragas.
# Synthesized contexts from session transcripts can be 100k+ chars;
# truncating prevents max_tokens errors during Ragas scoring.
MAX_CONTEXT_CHARS: int = 1_000

# Maximum characters for the response field sent to Ragas.
# Implement phase output can contain thousands of lines of raw code/JSON;
# Ragas faithfulness decomposes this into individual statements, which explodes.
MAX_RESPONSE_CHARS: int = 2_000

# Maximum number of retrieved context chunks sent to Ragas.
MAX_CONTEXT_CHUNKS: int = 10

# Maximum number of reference doc chunks to include.
MAX_REFERENCE_CHUNKS: int = 10

# Maximum characters per reference doc chunk.
MAX_REFERENCE_CHARS: int = 1_000

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


def _extract_response_summary(sample: EvalSample, implement_phase: PhaseResult) -> str:
    """Extract a concise response summary from structured phase data.

    Prefers structured fields (triage approach, plan tasks, implement
    deviations/commits) over raw implement output.  Falls back to
    truncated ``implement_phase.output`` when no structured data exists.

    The result is capped at :data:`MAX_RESPONSE_CHARS`.
    """
    parts: list[str] = []

    # Collect triage approach (latest generation only, consistent with _find_phase)
    latest_triage = _find_phase(sample, "triage")
    if latest_triage and latest_triage.output_structured:
        approach = latest_triage.output_structured.get("approach", "")
        if isinstance(approach, str) and approach:
            parts.append(f"Approach: {approach}")

    # Collect plan task descriptions (latest generation only)
    latest_plan = _find_phase(sample, "plan")
    if latest_plan and latest_plan.output_structured:
        tasks = latest_plan.output_structured.get("tasks", [])
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict):
                    description = task.get("description", "")
                    if description:
                        parts.append(f"Task: {description}")

    # Collect implement deviations and commit messages
    if implement_phase.output_structured:
        deviations = implement_phase.output_structured.get("deviations", [])
        if isinstance(deviations, list):
            for deviation in deviations:
                if isinstance(deviation, str) and deviation:
                    parts.append(f"Deviation: {deviation}")

        commits = implement_phase.output_structured.get("commits", [])
        if isinstance(commits, list):
            for commit in commits:
                if isinstance(commit, dict):
                    message = commit.get("message", "")
                    if message:
                        parts.append(f"Commit: {message}")

    if parts:
        summary = "\n".join(parts)
        return truncate_for_ragas(summary, max_chars=MAX_RESPONSE_CHARS)

    # Fallback: truncated raw output
    return truncate_for_ragas(implement_phase.output, max_chars=MAX_RESPONSE_CHARS)


def select_relevant_chunks(
    query: str,
    chunks: Sequence[DocChunk],
    top_k: int = MAX_REFERENCE_CHUNKS,
) -> list[DocChunk]:
    """Select the most relevant doc chunks by keyword overlap with *query*.

    Scores each chunk by the number of shared tokens (case-insensitive
    word split) with the query.  Returns the top *top_k* chunks, each
    truncated to :data:`MAX_REFERENCE_CHARS`.
    """
    if not chunks:
        return []

    query_tokens = set(query.lower().split())

    scored: list[tuple[int, int, DocChunk]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = set(chunk.text.lower().split())
        overlap = len(query_tokens & chunk_tokens)
        scored.append((overlap, idx, chunk))

    # Sort by overlap descending, then by original order for ties
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected = [(overlap, idx, chunk) for overlap, idx, chunk in scored[:top_k] if overlap > 0]

    return [
        DocChunk(
            text=truncate_for_ragas(chunk.text, max_chars=MAX_REFERENCE_CHARS),
            source_file=chunk.source_file,
            domain=chunk.domain,
        )
        for _overlap, _idx, chunk in selected
    ]


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
        # Cap retrieved contexts at MAX_CONTEXT_CHUNKS
        contexts = contexts[:MAX_CONTEXT_CHUNKS]
        if not contexts:
            continue
        user_input = _extract_question(sample)
        response = _extract_response_summary(sample, implement)
        reference = None
        if sample.ground_truth and sample.ground_truth.reference_answer:
            reference = sample.ground_truth.reference_answer
        elif doc_chunks:
            # Per-sample chunk selection based on user_input
            selected = select_relevant_chunks(user_input, doc_chunks)
            reference = "\n\n".join(chunk.text for chunk in selected) if selected else None
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


class InstructorSilentZeroError(RuntimeError):
    """Raised when the instructor#1658 silent-zero bug is detected for the Google provider.

    When ``instructor`` fails to parse Google's structured output it silently
    returns a Pydantic model with default field values (``value=0.0``,
    ``reason=None``) instead of raising a ``ValidationError``.  Raising this
    exception lets the existing per-session error handler log a warning and
    skip the session so the silent zero does not pollute the metric average.

    See: https://github.com/instructor-ai/instructor/issues/1658
    """


def is_instructor_silent_zero(result: object, provider: str) -> bool:
    """Detect the instructor#1658 silent-zero bug for the Google provider.

    When ``instructor`` fails to parse Google's structured output it silently
    returns a Pydantic model with default field values (``value=0.0``,
    ``reason=None``) instead of raising a ``ValidationError``.

    Returns ``True`` only when **all** of the following hold:

    - *provider* is ``"google"``
    - *result* is a structured object (not a plain ``float``)
    - ``result.value`` is exactly ``0.0``
    - ``result.reason`` is ``None`` or an empty string

    Args:
        result: The value returned by ``ragas_metric.ascore()``.
        provider: The LLM provider string from ``MetricConfig.llm_provider``.
    """
    if provider != "google":
        return False
    if isinstance(result, float):
        return False
    value = getattr(result, "value", None)
    reason = getattr(result, "reason", None)
    return value == 0.0 and not reason


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
