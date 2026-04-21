"""Faithfulness metric using Ragas 0.4 collections API.

Measures whether the generated response is faithful to the retrieved contexts
(i.e., not hallucinating beyond what the contexts provide).

EXPERIMENTAL: This metric was designed for natural language answers, not code.
Scores may be noisy for agentic code-generation sessions.
"""

import asyncio
import logging

from raki.adapters.redact import redact_sensitive
from raki.metrics.protocol import MetricConfig
from raki.metrics.ragas.adapter import detect_context_source, is_max_tokens_error, to_ragas_rows
from raki.metrics.ragas.async_utils import run_async
from raki.metrics.ragas.llm_setup import JudgeLogger, create_ragas_llm
from raki.model import EvalDataset
from raki.model.report import MetricResult

logger = logging.getLogger(__name__)


class FaithfulnessMetric:
    """Ragas-backed faithfulness metric satisfying the Metric protocol.

    Uses the public Ragas 0.4 collections API with ascore() keyword args.
    Marked as experimental -- designed for NL answers, not code.
    """

    name: str = "faithfulness"
    requires_ground_truth: bool = False
    requires_llm: bool = True
    higher_is_better: bool = True
    display_format: str = "score"
    display_name: str = "Faithfulness"
    description: str = "Whether the response is faithful to retrieved contexts"
    experimental: bool = True

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
            Faithfulness as RagasFaithfulness,
        )

        llm = create_ragas_llm(config)
        ragas_metric = RagasFaithfulness(llm=llm)

        judge_logger: JudgeLogger | None = None
        if config.judge_log_path is not None:
            judge_logger = JudgeLogger(config.judge_log_path, config.project_root)

        sample_scores: dict[str, float] = {}
        scores: list[float] = []
        max_tokens_failures: list[str] = []

        async def score_all():
            semaphore = asyncio.Semaphore(config.batch_size)

            async def score_one(row):
                async with semaphore:
                    try:
                        result = await ragas_metric.ascore(
                            user_input=row.user_input,
                            response=row.response,
                            retrieved_contexts=row.retrieved_contexts,
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
                        logger.warning(
                            "faithfulness scoring failed for session %s: %s",
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
        if context_source is not None:
            details["context_source"] = context_source

        return MetricResult(
            name=self.name,
            score=mean_score,
            details=details,
            sample_scores=sample_scores,
        )
