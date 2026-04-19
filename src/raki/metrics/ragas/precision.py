"""Context precision metric using Ragas 0.4 ContextPrecisionWithReference.

Measures how relevant the retrieved contexts are to answering the user's question,
given a reference answer. Higher is better.
"""

import asyncio
import logging

from raki.adapters.redact import redact_sensitive
from raki.metrics.protocol import MetricConfig
from raki.metrics.ragas.adapter import to_ragas_rows
from raki.metrics.ragas.async_utils import run_async
from raki.metrics.ragas.llm_setup import JudgeLogger, create_ragas_llm
from raki.model import EvalDataset
from raki.model.report import MetricResult

logger = logging.getLogger(__name__)


class ContextPrecisionMetric:
    """Ragas-backed context precision metric satisfying the Metric protocol."""

    name: str = "context_precision"
    requires_ground_truth: bool = True
    requires_llm: bool = True
    higher_is_better: bool = True
    display_format: str = "score"
    display_name: str = "Context precision"
    description: str = "Relevance of retrieved contexts to the question"

    def compute(
        self,
        dataset: EvalDataset,
        config: MetricConfig,
    ) -> MetricResult:
        rows = to_ragas_rows(dataset)
        rows_with_ref = [row for row in rows if row.reference is not None]
        if not rows_with_ref:
            return MetricResult(name=self.name, score=0.0, details={"skipped": "no ground truth"})

        from ragas.metrics.collections import (  # ty: ignore[unresolved-import]
            ContextPrecisionWithReference,
        )

        llm = create_ragas_llm(config)
        ragas_metric = ContextPrecisionWithReference(llm=llm)

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
                            retrieved_contexts=row.retrieved_contexts,
                            reference=row.reference,
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
                            "context_precision scoring failed for session %s: %s",
                            row.session_id,
                            safe_error,
                        )
                        if judge_logger:
                            judge_logger.log(self.name, row.user_input, -1.0, safe_error)

            await asyncio.gather(*(score_one(row) for row in rows_with_ref))

        run_async(score_all())

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            score=mean_score,
            details={
                "samples_scored": len(scores),
                "samples_skipped": len(rows_with_ref) - len(scores),
            },
            sample_scores=sample_scores,
        )
