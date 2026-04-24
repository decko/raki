"""Shared async scoring loop extracted from the four Ragas metric files.

All four metrics (faithfulness, precision, recall, relevancy) duplicate
identical error-handling logic for InstructorSilentZeroError, max_tokens
failures, and per-session score accumulation. This module consolidates
that logic so each metric only supplies the metric-specific score_fn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from raki.adapters.redact import redact_sensitive
from raki.metrics.ragas.adapter import (
    InstructorSilentZeroError,
    RagasRow,
    is_instructor_silent_zero,
    is_max_tokens_error,
)
from raki.metrics.ragas.llm_setup import JudgeLogger
from raki.model.report import MetricResult

logger = logging.getLogger(__name__)


@dataclass
class ScoringState:
    """Accumulates results and failures from a parallel scoring run.

    Attributes:
        scores: Successful float scores from each scored row.
        sample_scores: Maps session_id to its score (for per-sample reporting).
        max_tokens_failures: Session IDs that failed due to max_tokens errors.
        silent_zero_failures: Session IDs that hit the instructor#1658 silent-zero bug.
    """

    scores: list[float] = field(default_factory=list)
    sample_scores: dict[str, float] = field(default_factory=dict)
    max_tokens_failures: list[str] = field(default_factory=list)
    silent_zero_failures: list[str] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        """Average of all successful scores, or 0.0 if none."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


async def score_rows(
    rows: list[RagasRow],
    score_fn: Callable[[RagasRow], Awaitable[object]],
    metric_name: str,
    llm_provider: str,
    batch_size: int,
    judge_logger: JudgeLogger | None,
) -> ScoringState:
    """Score all rows concurrently, collecting successes and categorised failures.

    Calls ``score_fn(row)`` for each row with a semaphore limiting concurrency
    to ``batch_size``. Handles three exception classes specially:

    - ``max_tokens`` errors (provider output token limit) → max_tokens_failures
    - :class:`InstructorSilentZeroError` (Google #1658 bug) → silent_zero_failures
    - All other exceptions are logged but do not populate failure lists

    Args:
        rows: The rows to score.
        score_fn: Async callable that invokes ``ragas_metric.ascore()`` for a row.
        metric_name: Name of the metric, used in warning messages and judge log.
        llm_provider: LLM provider string from ``MetricConfig.llm_provider``,
            used for silent-zero detection.
        batch_size: Maximum number of concurrent ``score_fn`` calls.
        judge_logger: Optional audit logger for all judge calls.

    Returns:
        :class:`ScoringState` with accumulated scores and failure lists.
    """
    state = ScoringState()
    semaphore = asyncio.Semaphore(batch_size)

    async def score_one(row: RagasRow) -> None:
        async with semaphore:
            try:
                result = await score_fn(row)
                if is_instructor_silent_zero(result, llm_provider):
                    raise InstructorSilentZeroError(
                        f"instructor#1658: Google provider returned silent 0.0 "
                        f"for session {row.session_id}; treating as failure to "
                        "avoid polluting metric average"
                    )
                score = result if isinstance(result, float) else result.value  # ty: ignore[unresolved-attribute]
                state.scores.append(score)
                state.sample_scores[row.session_id] = score
                reason = None if isinstance(result, float) else result.reason  # ty: ignore[unresolved-attribute]
                if judge_logger:
                    judge_logger.log(metric_name, row.user_input, score, reason)
            except Exception as exc:
                safe_error = redact_sensitive(f"ERROR: {exc}")
                if is_max_tokens_error(exc):
                    state.max_tokens_failures.append(row.session_id)
                elif isinstance(exc, InstructorSilentZeroError):
                    state.silent_zero_failures.append(row.session_id)
                logger.warning(
                    "%s scoring failed for session %s: %s",
                    metric_name,
                    row.session_id,
                    safe_error,
                )
                if judge_logger:
                    judge_logger.log(metric_name, row.user_input, -1.0, safe_error)

    await asyncio.gather(*(score_one(row) for row in rows))
    return state


def build_max_tokens_result(metric_name: str, state: ScoringState) -> MetricResult | None:
    """Return a ``score=None`` MetricResult when all failures were max_tokens errors.

    Returns ``None`` when there are any successful scores or when there are no
    max_tokens failures, so callers can use it as a guard:

    .. code-block:: python

        if result := build_max_tokens_result(self.name, state):
            return result

    Args:
        metric_name: Name passed to :class:`MetricResult`.
        state: The :class:`ScoringState` from :func:`score_rows`.
    """
    if not state.scores and state.max_tokens_failures:
        return MetricResult(
            name=metric_name,
            score=None,
            details={
                "skipped": "max_tokens: all sessions exceeded output token limit",
                "max_tokens_sessions": len(state.max_tokens_failures),
            },
        )
    return None


def build_silent_zero_result(metric_name: str, state: ScoringState) -> MetricResult | None:
    """Return a ``score=None`` MetricResult when all failures were silent-zero errors.

    Returns ``None`` when there are any successful scores or when there are no
    silent-zero failures, so callers can use it as a guard:

    .. code-block:: python

        if result := build_silent_zero_result(self.name, state):
            return result

    Args:
        metric_name: Name passed to :class:`MetricResult`.
        state: The :class:`ScoringState` from :func:`score_rows`.
    """
    if not state.scores and state.silent_zero_failures:
        return MetricResult(
            name=metric_name,
            score=None,
            details={
                "skipped": (
                    "instructor#1658: Google provider returned only silent 0.0 scores; "
                    "possible structured-output parsing failures"
                ),
                "silent_zero_sessions": len(state.silent_zero_failures),
            },
        )
    return None


def enrich_details_with_failures(details: dict, state: ScoringState) -> None:
    """Add failure-count keys to *details* in-place.

    Appends ``max_tokens_sessions`` when there are max_tokens failures.
    Appends ``silent_zero_sessions`` and ``silent_zero_warning`` when there
    are instructor#1658 silent-zero failures.

    Args:
        details: The metric details dict to mutate.
        state: The :class:`ScoringState` from :func:`score_rows`.
    """
    if state.max_tokens_failures:
        details["max_tokens_sessions"] = len(state.max_tokens_failures)
    if state.silent_zero_failures:
        details["silent_zero_sessions"] = len(state.silent_zero_failures)
        details["silent_zero_warning"] = (
            f"instructor#1658: {len(state.silent_zero_failures)} session(s) returned silent 0.0 "
            "from Google provider and were excluded from the score"
        )
