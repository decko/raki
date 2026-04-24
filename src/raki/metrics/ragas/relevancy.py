"""Answer relevancy metric using Ragas 0.4 collections API.

Measures how relevant the generated response is to the user's question.
Requires both an LLM (for generation) and embeddings (for similarity scoring).

EXPERIMENTAL: This metric was designed for natural language answers, not code.
Scores may be noisy for agentic code-generation sessions.
"""

import asyncio
import logging

from raki.adapters.redact import redact_sensitive
from raki.metrics.protocol import MetricConfig
from raki.metrics.ragas.adapter import (
    InstructorSilentZeroError,
    detect_context_source,
    is_instructor_silent_zero,
    is_max_tokens_error,
    to_ragas_rows,
)
from raki.metrics.ragas.async_utils import run_async
from raki.metrics.ragas.llm_setup import JudgeLogger, create_ragas_embeddings, create_ragas_llm
from raki.model import EvalDataset
from raki.model.report import MetricResult

logger = logging.getLogger(__name__)


class AnswerRelevancyMetric:
    """Ragas-backed answer relevancy metric satisfying the Metric protocol.

    Uses the public Ragas 0.4 collections API with ascore() keyword args.
    Marked as experimental -- designed for NL answers, not code.
    """

    name: str = "answer_relevancy"
    requires_ground_truth: bool = False
    requires_llm: bool = True
    higher_is_better: bool = True
    display_format: str = "score"
    display_name: str = "Answer relevancy"
    description: str = "Relevance of the generated response to the question"
    experimental: bool = True
    rationale: str = (
        "Answer relevancy measures whether the agent's output actually addresses the user's "
        "question. An LLM generates synthetic questions from the response and compares them "
        "to the original using embedding similarity. Low relevancy indicates the agent is "
        "going off-track — answering a different question or producing unrelated output. "
        "This is typically a prompt or routing issue rather than a retrieval problem. "
        "Experimental: multi-step agentic workflows may address the question indirectly, "
        "yielding lower scores without indicating a real problem."
    )

    def compute(
        self,
        dataset: EvalDataset,
        config: MetricConfig,
    ) -> MetricResult:
        rows = to_ragas_rows(dataset)
        if not rows:
            return MetricResult(
                name=self.name, score=None, details={"skipped": "no retrieval context"}
            )

        # Guard: if all rows have empty retrieved_contexts, return N/A
        rows_with_contexts = [row for row in rows if row.retrieved_contexts]
        if not rows_with_contexts:
            return MetricResult(
                name=self.name, score=None, details={"skipped": "no retrieval context"}
            )

        from ragas.metrics.collections import (  # ty: ignore[unresolved-import]
            AnswerRelevancy as RagasAnswerRelevancy,
        )

        llm = create_ragas_llm(config)
        embeddings = create_ragas_embeddings()
        ragas_metric = RagasAnswerRelevancy(llm=llm, embeddings=embeddings)

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
                            response=row.response,
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
                            "answer_relevancy scoring failed for session %s: %s",
                            row.session_id,
                            safe_error,
                        )
                        if judge_logger:
                            judge_logger.log(self.name, row.user_input, -1.0, safe_error)

            await asyncio.gather(*(score_one(row) for row in rows))

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

        # Determine context_source from the dataset samples
        context_source = detect_context_source(dataset)

        details: dict = {
            "samples_scored": len(scores),
            "samples_skipped": len(rows) - len(scores),
            "experimental": True,
            "caveat": (
                "Designed for NL answers, not code -- scores may be noisy for agentic sessions"
            ),
        }
        if max_tokens_failures:
            details["max_tokens_sessions"] = len(max_tokens_failures)
        if silent_zero_failures:
            details["silent_zero_sessions"] = len(silent_zero_failures)
            details["silent_zero_warning"] = (
                f"instructor#1658: {len(silent_zero_failures)} session(s) returned silent 0.0 "
                "from Google provider and were excluded from the score"
            )
        if context_source is not None:
            details["context_source"] = context_source

        return MetricResult(
            name=self.name,
            score=mean_score,
            details=details,
            sample_scores=sample_scores,
        )
