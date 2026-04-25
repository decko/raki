Add ``"litellm"`` as a supported LLM provider for Ragas judge metrics.

Set ``llm_provider: litellm`` in your ``MetricConfig`` to route model calls
through LiteLLM, enabling any model supported by LiteLLM (OpenAI, Bedrock,
Cohere, etc.) as a judge. Token usage is tracked at the ``litellm.acompletion``
module level and reported in the standard judge-cost summary.

Install the optional dependency with ``pip install raki[litellm]``.
