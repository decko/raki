"""LLM setup and judge logging for Ragas metrics.

Uses Ragas 0.4 llm_factory with configurable LLM client:
- vertex-anthropic (default): AsyncAnthropicVertex for Vertex AI
- anthropic: AsyncAnthropic for direct Anthropic API
- google: Google GenAI client via Vertex AI
- litellm: LiteLLM module via Ragas's LiteLLMAdapter
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from raki.adapters.redact import redact_sensitive
from raki.metrics.protocol import LLMProvider, MetricConfig

if TYPE_CHECKING:
    from raki.metrics.protocol import TokenAccumulator

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = get_args(LLMProvider)


def patch_client_for_token_tracking(client: object, accumulator: TokenAccumulator) -> None:
    """Monkey-patch an Anthropic client to track token usage.

    Wraps ``client.messages.create`` so that every call increments the
    accumulator with the response's ``usage.input_tokens`` and
    ``usage.output_tokens``.  The response is returned unmodified.

    Only intended for Anthropic-style clients (AsyncAnthropic /
    AsyncAnthropicVertex) whose ``messages.create`` is an async method.
    """
    original_create = client.messages.create  # ty: ignore[unresolved-attribute]

    @functools.wraps(original_create)
    async def tracked_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        response = await original_create(*args, **kwargs)
        if hasattr(response, "usage"):
            accumulator.input_tokens += response.usage.input_tokens
            accumulator.output_tokens += response.usage.output_tokens
        accumulator.calls += 1
        return response

    client.messages.create = tracked_create  # ty: ignore[unresolved-attribute]


def patch_litellm_for_token_tracking(litellm_module: object, accumulator: TokenAccumulator) -> None:
    """Monkey-patch the litellm module to track token usage.

    Wraps ``litellm.acompletion`` so that every call increments the accumulator
    with the response's ``usage.prompt_tokens`` and ``usage.completion_tokens``.
    The response is returned unmodified.

    Ragas passes the ``litellm`` module itself as the client to
    ``LiteLLMAdapter``, so patching at module level is the correct approach.
    """
    original_acompletion = litellm_module.acompletion  # ty: ignore[unresolved-attribute]

    @functools.wraps(original_acompletion)
    async def tracked_acompletion(*args, **kwargs):  # type: ignore[no-untyped-def]
        response = await original_acompletion(*args, **kwargs)
        if hasattr(response, "usage") and response.usage is not None:
            accumulator.input_tokens += response.usage.prompt_tokens
            accumulator.output_tokens += response.usage.completion_tokens
        accumulator.calls += 1
        return response

    litellm_module.acompletion = tracked_acompletion  # ty: ignore[unresolved-attribute]


def create_ragas_llm(config: MetricConfig):
    """Create a Ragas LLM using the 0.4 llm_factory.

    Dispatches on ``config.llm_provider``:

    - ``vertex-anthropic`` (default) -- uses ``AsyncAnthropicVertex``
    - ``anthropic`` -- uses ``AsyncAnthropic`` (direct Anthropic API)
    - ``google`` -- uses ``google.genai.Client`` via Vertex AI
    - ``litellm`` -- uses the ``litellm`` module via Ragas's ``LiteLLMAdapter``

    Defers ragas and provider-specific imports so this module can be imported
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
    elif config.llm_provider == "google":
        import os

        from google import genai  # ty: ignore[unresolved-import]

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEXAI_PROJECT")
        if not project:
            raise ValueError(
                "Google provider requires GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT environment variable"
            )
        location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)

        from ragas.llms import llm_factory  # ty: ignore[unresolved-import]

        llm = llm_factory(
            config.llm_model,
            provider="google",
            client=client,
            temperature=config.temperature,
            max_tokens=4096,
        )
        llm.model_args.pop("top_p", None)
        return llm
    elif config.llm_provider == "litellm":
        import litellm  # ty: ignore[unresolved-import]

        if config.token_accumulator is not None:
            patch_litellm_for_token_tracking(litellm, config.token_accumulator)

        from ragas.llms import llm_factory  # ty: ignore[unresolved-import]

        llm = llm_factory(
            config.llm_model,
            provider="litellm",
            client=litellm,
            temperature=config.temperature,
            max_tokens=4096,
        )
        llm.model_args.pop("top_p", None)
        return llm
    else:
        supported_list = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown LLM provider: '{config.llm_provider}'. Supported providers: {supported_list}"
        )

    if config.token_accumulator is not None:
        patch_client_for_token_tracking(client, config.token_accumulator)

    from ragas.llms import llm_factory  # ty: ignore[unresolved-import]

    llm = llm_factory(
        config.llm_model,
        provider="anthropic",
        client=client,
        temperature=config.temperature,
        max_tokens=4096,
    )
    llm.model_args.pop("top_p", None)
    return llm


def create_ragas_embeddings(config: MetricConfig):
    """Create embeddings for answer_relevancy metric.

    Dispatches on ``config.llm_provider``:

    - ``vertex-anthropic``, ``anthropic``, ``google`` -- uses ``GoogleEmbeddings``
      constructed directly with ``use_vertex=True`` and a pre-configured
      ``genai.Client`` for Vertex AI.  This bypasses ``embedding_factory()``
      which does not forward ``use_vertex`` to the constructor, causing
      ``_resolve_client()`` to take the Gemini path and discard the
      pre-configured client (see #106).

    - ``litellm`` -- uses Ragas's built-in ``LiteLLMEmbeddings`` with model
      ``text-embedding-3-small`` (OpenAI-compatible, routed via LiteLLM).

    Resolves project/location from GOOGLE_CLOUD_PROJECT (or VERTEXAI_PROJECT)
    and VERTEXAI_LOCATION environment variables (Google providers only).

    Defers imports so this module can be imported without dependencies installed.
    """
    if config.llm_provider == "litellm":
        from ragas.embeddings import LiteLLMEmbeddings  # ty: ignore[unresolved-import]

        return LiteLLMEmbeddings(model="text-embedding-3-small")

    import os

    from google import genai  # ty: ignore[unresolved-import]
    from ragas.embeddings.google_provider import GoogleEmbeddings  # ty: ignore[unresolved-import]

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEXAI_PROJECT")
    if not project:
        raise ValueError(
            "Embeddings require GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT environment variable"
        )
    location = os.environ.get("VERTEXAI_LOCATION", "us-central1")

    client = genai.Client(vertexai=True, project=project, location=location)

    return GoogleEmbeddings(
        client=client,
        model="text-embedding-005",
        use_vertex=True,
    )


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
