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
from raki.metrics.ragas.adapter import to_ragas_rows
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
    experimental: bool = True

    def compute(
        self,
        dataset: EvalDataset,
        config: MetricConfig,
    ) -> MetricResult:
        rows = to_ragas_rows(dataset)
        if not rows:
            return MetricResult(name=self.name, score=0.0, details={"skipped": "no samples"})

        from ragas.metrics.collections import (  # ty: ignore[unresolved-import]
            AnswerRelevancy as RagasAnswerRelevancy,
        )

        llm = create_ragas_llm(config)
        embeddings = create_ragas_embeddings()
        ragas_metric = RagasAnswerRelevancy(llm=llm, embeddings=embeddings)

        judge_logger: JudgeLogger | None = None
        if config.judge_log_path is not None:
            judge_logger = JudgeLogger(config.judge_log_path)

        sample_scores: dict[str, float] = {}
        scores: list[float] = []

        async def score_all():
            semaphore = asyncio.Semaphore(config.batch_size)

            async def score_one(row):
                async with semaphore:
                    try:
                        result = await ragas_metric.ascore(
                            user_input=row.user_input,
                            response=row.response,
                        )
                        score = result if isinstance(result, float) else result.value
                        scores.append(score)
                        sample_scores[row.session_id] = score
                        reason = None if isinstance(result, float) else result.reason
                        if judge_logger:
                            judge_logger.log(self.name, row.user_input, score, reason)
                    except Exception as exc:
                        safe_error = redact_sensitive(f"ERROR: {exc}")
                        logger.warning(
                            "answer_relevancy scoring failed for session %s: %s",
                            row.session_id,
                            safe_error,
                        )
                        if judge_logger:
                            judge_logger.log(self.name, row.user_input, -1.0, safe_error)

            await asyncio.gather(*(score_one(row) for row in rows))

        run_async(score_all())

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            score=mean_score,
            details={
                "samples_scored": len(scores),
                "samples_skipped": len(rows) - len(scores),
                "experimental": True,
                "caveat": (
                    "Designed for NL answers, not code -- scores may be noisy for agentic sessions"
                ),
            },
            sample_scores=sample_scores,
        )
