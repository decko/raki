"""Context recall metric using Ragas 0.4 ContextRecall.

Measures how well the retrieved contexts cover the information needed
to answer the question, given a reference answer. Higher is better.
"""

import logging

from raki.metrics.protocol import MetricConfig
from raki.metrics.ragas.adapter import (
    to_ragas_rows,
)
from raki.metrics.ragas.async_utils import run_async
from raki.metrics.ragas.llm_setup import JudgeLogger, create_ragas_llm
from raki.metrics.ragas._scoring_loop import (
    ScoringState,
    build_max_tokens_result,
    build_silent_zero_result,
    enrich_details_with_failures,
    score_rows,
)
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

        async def score_fn(row):
            return await ragas_metric.ascore(
                user_input=row.user_input,
                retrieved_contexts=row.retrieved_contexts,
                reference=row.reference,
            )

        state: ScoringState = run_async(
            score_rows(
                rows=rows_with_ref,
                score_fn=score_fn,
                metric_name=self.name,
                llm_provider=config.llm_provider,
                batch_size=config.batch_size,
                judge_logger=judge_logger,
            )
        )

        # If all failures were max_tokens errors, return score=None
        if result := build_max_tokens_result(self.name, state):
            return result

        # If all failures were instructor#1658 silent-zero errors, return score=None
        if result := build_silent_zero_result(self.name, state):
            return result

        details: dict = {
            "samples_scored": len(state.scores),
            "samples_skipped": len(rows_with_ref) - len(state.scores),
        }
        enrich_details_with_failures(details, state)
        return MetricResult(
            name=self.name,
            score=state.mean_score,
            details=details,
            sample_scores=state.sample_scores,
        )
