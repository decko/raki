"""Answer relevancy metric using Ragas 0.4 collections API.

Measures how relevant the generated response is to the user's question.
Requires both an LLM (for generation) and embeddings (for similarity scoring).

EXPERIMENTAL: This metric was designed for natural language answers, not code.
Scores may be noisy for agentic code-generation sessions.
"""

import logging

from raki.metrics.protocol import MetricConfig
from raki.metrics.ragas.adapter import (
    detect_context_source,
    to_ragas_rows,
)
from raki.metrics.ragas.async_utils import run_async
from raki.metrics.ragas.llm_setup import JudgeLogger, create_ragas_embeddings, create_ragas_llm
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
        embeddings = create_ragas_embeddings(config)
        ragas_metric = RagasAnswerRelevancy(llm=llm, embeddings=embeddings)

        judge_logger: JudgeLogger | None = None
        if config.judge_log_path is not None:
            judge_logger = JudgeLogger(config.judge_log_path, config.project_root)

        async def score_fn(row):
            return await ragas_metric.ascore(
                user_input=row.user_input,
                response=row.response,
            )

        state: ScoringState = run_async(
            score_rows(
                rows=rows,
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

        # Determine context_source from the dataset samples
        context_source = detect_context_source(dataset)

        details: dict = {
            "samples_scored": len(state.scores),
            "samples_skipped": len(rows) - len(state.scores),
            "experimental": True,
            "caveat": (
                "Designed for NL answers, not code -- scores may be noisy for agentic sessions"
            ),
        }
        enrich_details_with_failures(details, state)
        if context_source is not None:
            details["context_source"] = context_source

        return MetricResult(
            name=self.name,
            score=state.mean_score,
            details=details,
            sample_scores=state.sample_scores,
        )
