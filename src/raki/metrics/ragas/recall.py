"""Context recall metric using Ragas 0.4 ContextRecall.

Measures how well the retrieved contexts cover the information needed
to answer the question, given a reference answer. Higher is better.
"""

import asyncio
import logging

from raki.adapters.redact import redact_sensitive
from raki.metrics.protocol import MetricConfig
from raki.metrics.ragas.adapter import (
    InstructorSilentZeroError,
    is_instructor_silent_zero,
    is_max_tokens_error,
    to_ragas_rows,
)
from raki.metrics.ragas.async_utils import run_async
from raki.metrics.ragas.llm_setup import JudgeLogger, create_ragas_llm
from raki.model import EvalDataset
from raki.model.report import MetricResult

logger = logging.getLogger(__name__)


class ContextRecallMetric:
    """Ragas-backed context recall metric satisfying the Metric protocol."""

    name: str = "context_recall"
    requires_ground_truth: bool = True
    requires_llm: bool = True
    higher_is_better: bool = True
    display_format: str = "score"
    display_name: str = "Context recall"
    description: str = "Coverage of needed information in retrieved contexts"
    rationale: str = (
        "Context recall measures how much of the needed information the retriever successfully "
        "found. Low recall means the agent is missing critical knowledge — either it does not "
        "exist in the knowledge base, or search is failing to surface it. Requires ground truth "
        "to define what information is 'needed'. Complementary to context_precision: high "
        "precision with low recall means the retriever is selective but incomplete; low "
        "precision with high recall means it over-retrieves but does find the relevant content."
    )

    def compute(
        self,
        dataset: EvalDataset,
        config: MetricConfig,
    ) -> MetricResult:
        rows = to_ragas_rows(dataset, doc_chunks=config.doc_chunks or None)
        rows_with_ref = [row for row in rows if row.reference is not None]
        if not rows_with_ref:
            return MetricResult(name=self.name, score=None, details={"skipped": "no ground truth"})

        from ragas.metrics.collections import (  # ty: ignore[unresolved-import]
            ContextRecall,
        )

        llm = create_ragas_llm(config)
        ragas_metric = ContextRecall(llm=llm)

        judge_logger: JudgeLogger | None = None
        if config.judge_log_path is not None:
            judge_logger = JudgeLogger(config.judge_log_path, config.project_root)

        sample_scores: dict[str, float] = {}
        scores: list[float] = []
        max_tokens_failures: list[str] = []
        silent_zero_failures: list[str] = []

        async def score_all():
            semaphore = asyncio.Semaphore(config.batch_size)

            async def score_one(row):
                async with semaphore:
                    try:
                        result = await ragas_metric.ascore(
                            user_input=row.user_input,
                            retrieved_contexts=row.retrieved_contexts,
                            reference=row.reference,
                        )
                        if is_instructor_silent_zero(result, config.llm_provider):
                            raise InstructorSilentZeroError(
                                f"instructor#1658: Google provider returned silent 0.0 "
                                f"for session {row.session_id}; treating as failure to "
                                "avoid polluting metric average"
                            )
                        score = result if isinstance(result, float) else result.value
                        scores.append(score)
                        sample_scores[row.session_id] = score
                        reason = None if isinstance(result, float) else result.reason
                        if judge_logger:
                            judge_logger.log(self.name, row.user_input, score, reason)
                    except Exception as exc:
                        safe_error = redact_sensitive(f"ERROR: {exc}")
                        if is_max_tokens_error(exc):
                            max_tokens_failures.append(row.session_id)
                        elif isinstance(exc, InstructorSilentZeroError):
                            silent_zero_failures.append(row.session_id)
                        logger.warning(
                            "context_recall scoring failed for session %s: %s",
                            row.session_id,
                            safe_error,
                        )
                        if judge_logger:
                            judge_logger.log(self.name, row.user_input, -1.0, safe_error)

            await asyncio.gather(*(score_one(row) for row in rows_with_ref))

        run_async(score_all())

        # If all failures were max_tokens errors, return score=None
        if not scores and max_tokens_failures:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "skipped": "max_tokens: all sessions exceeded output token limit",
                    "max_tokens_sessions": len(max_tokens_failures),
                },
            )

        # If all failures were instructor#1658 silent-zero errors, return score=None
        if not scores and silent_zero_failures:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "skipped": (
                        "instructor#1658: Google provider returned only silent 0.0 scores; "
                        "possible structured-output parsing failures"
                    ),
                    "silent_zero_sessions": len(silent_zero_failures),
                },
            )

        mean_score = sum(scores) / len(scores) if scores else 0.0
        details: dict = {
            "samples_scored": len(scores),
            "samples_skipped": len(rows_with_ref) - len(scores),
        }
        if max_tokens_failures:
            details["max_tokens_sessions"] = len(max_tokens_failures)
        if silent_zero_failures:
            details["silent_zero_sessions"] = len(silent_zero_failures)
            details["silent_zero_warning"] = (
                f"instructor#1658: {len(silent_zero_failures)} session(s) returned silent 0.0 "
                "from Google provider and were excluded from the score"
            )
        return MetricResult(
            name=self.name,
            score=mean_score,
            details=details,
            sample_scores=sample_scores,
        )
