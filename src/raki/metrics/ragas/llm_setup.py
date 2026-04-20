"""LLM setup and judge logging for Ragas metrics.

Uses Ragas 0.4 llm_factory with configurable Anthropic client:
- vertex-anthropic (default): AsyncAnthropicVertex for Vertex AI
- anthropic: AsyncAnthropic for direct Anthropic API
"""

import json
import logging
from pathlib import Path
from typing import get_args

from raki.adapters.redact import redact_sensitive
from raki.metrics.protocol import LLMProvider, MetricConfig

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = get_args(LLMProvider)


def create_ragas_llm(config: MetricConfig):
    """Create a Ragas LLM using the 0.4 llm_factory.

    Dispatches on ``config.llm_provider``:

    - ``vertex-anthropic`` (default) -- uses ``AsyncAnthropicVertex``
    - ``anthropic`` -- uses ``AsyncAnthropic`` (direct Anthropic API)

    Defers ragas and anthropic imports so this module can be imported
    without those packages installed.

    Raises:
        ValueError: If ``config.llm_provider`` is not a supported provider.
    """
    if config.llm_provider == "vertex-anthropic":
        from anthropic import AsyncAnthropicVertex  # ty: ignore[unresolved-import]

        client = AsyncAnthropicVertex()
    elif config.llm_provider == "anthropic":
        from anthropic import AsyncAnthropic  # ty: ignore[unresolved-import]

        client = AsyncAnthropic()
    else:
        supported_list = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown LLM provider: '{config.llm_provider}'. Supported providers: {supported_list}"
        )

    from ragas.llms import llm_factory  # ty: ignore[unresolved-import]

    return llm_factory(
        config.llm_model,
        client=client,
        temperature=config.temperature,
    )


def create_ragas_embeddings():
    """Create embeddings for answer_relevancy metric.

    Uses VertexAIEmbeddings from langchain-google-vertexai for consistency
    with the Vertex AI LLM setup. The previous embedding_factory() defaulted
    to OpenAI, which was inconsistent with the Anthropic/Vertex AI LLM config.

    Defers imports so this module can be imported without dependencies installed.
    """
    from langchain_google_vertexai import VertexAIEmbeddings  # ty: ignore[unresolved-import]

    return VertexAIEmbeddings(model_name="text-embedding-005")


def _validate_judge_log_path(log_path: Path, project_root: Path | None = None) -> Path:
    """Validate that the judge log path is under the project root.

    Follows the same path traversal guard pattern used by the session schema
    adapter and manifest loader.

    Args:
        log_path: Path to the judge log file.
        project_root: Root directory to validate against. Falls back to
            Path.cwd() if not provided (for backward compatibility).
    """
    resolved = log_path.resolve()
    root = (project_root or Path.cwd()).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"judge_log_path escapes project root: {resolved} is not under {root}")
    return resolved


class JudgeLogger:
    """Logs all LLM judge calls to JSONL for audit.

    Each line is a JSON object with metric name, truncated and redacted
    user_input, score, and optional reason.
    """

    def __init__(self, log_path: Path, project_root: Path | None = None):
        self.log_path = _validate_judge_log_path(log_path, project_root)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        metric_name: str,
        user_input: str,
        result_value: float,
        result_reason: str | None,
    ) -> None:
        """Append a judge call record to the JSONL log file."""
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {
                        "metric": metric_name,
                        "user_input": redact_sensitive(user_input)[:200],
                        "score": result_value,
                        "reason": result_reason,
                    }
                )
                + "\n"
            )
