# Analytical Metrics Reference

Analytical metrics use an LLM judge (via Ragas) to evaluate retrieval quality. They require the `--judge` flag and LLM provider credentials.

## Prerequisites

Enable analytical metrics with `--judge`:

```bash
# Using Vertex AI Anthropic (default provider)
uv run raki run --manifest raki.yaml --judge

# Using direct Anthropic API
uv run raki run --manifest raki.yaml --judge --judge-provider anthropic

# Using Google AI
uv run raki run --manifest raki.yaml --judge --judge-provider google

# Using LiteLLM (any supported model, e.g. OpenAI)
uv run raki run --manifest raki.yaml --judge \
  --judge-provider litellm --judge-model gpt-4o
```

Provider options for `--judge-provider`:
- `vertex-anthropic` (default) -- Claude via Vertex AI
- `anthropic` -- direct Anthropic API (requires `ANTHROPIC_API_KEY`)
- `google` -- Google AI
- `litellm` -- any model via [LiteLLM](https://docs.litellm.ai/) (requires `raki[litellm]` and the appropriate provider credentials, e.g. `OPENAI_API_KEY`)

The default judge model is `claude-sonnet-4-6`. Override with `--judge-model`.

When using `litellm`, embeddings are served by `text-embedding-3-small` (OpenAI) via LiteLLM.
Set `OPENAI_API_KEY` or configure the LiteLLM proxy for a different embedding provider.

Install with all extras to get Ragas dependencies:

```bash
uv sync --python 3.12 --all-extras
```

Or install only what you need:

```bash
uv pip install raki[ragas,litellm]
```

> **Note:** The `ragas` extra pulls `scikit-network`, which requires a C++ compiler (`g++`).

## Context synthesis

RAKI synthesizes context for Ragas evaluation from session phase data. The agent's tool calls, retrieved documents, and phase outputs are assembled into `retrieved_contexts` and `response` fields that Ragas evaluates.

When the context source is inferred from phase data rather than explicit retrieval logs, metric names may appear with an `(inferred)` suffix in the report to indicate that context was reconstructed rather than directly captured.

## faithfulness -- Faithfulness (experimental)

**What it measures:** Whether the agent's output is grounded in the retrieved context -- i.e., the agent is not hallucinating beyond what the sources provide.

**What it tells you:** How closely the agent sticks to facts in its source material.

**What action it drives:** Inspect low-scoring sessions manually. Distinguish between genuine hallucination and legitimate multi-step reasoning before taking action.

**How it's computed:** Ragas Faithfulness metric. Uses LLM-as-judge to check each claim in the response against retrieved contexts.

**N/A conditions:** Returns `score=None` when no sessions have retrieval context.

**Experimental caveat:** This metric was designed for natural language answers, not code. Scores may be noisy for agentic code-generation sessions where the agent synthesizes across tool calls.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Output is well-grounded in context |
| Yellow | 0.50--0.79 | Some claims lack context support |
| Red | < 0.50 | Output frequently diverges from context |

---

## answer_relevancy -- Answer relevancy (experimental)

**What it measures:** How relevant the agent's output is to the original user query.

**What it tells you:** Whether the agent is addressing the actual question or going off-track.

**What action it drives:** Low relevancy often indicates a prompt or routing issue rather than a retrieval problem. Check whether the agent is interpreting the task correctly.

**How it's computed:** Ragas AnswerRelevancy metric. Uses LLM + embeddings to measure semantic similarity between the response and the question.

**N/A conditions:** Returns `score=None` when no sessions have retrieval context.

**Experimental caveat:** Multi-step agent workflows may address the question indirectly, lowering scores without indicating a real problem.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Output directly addresses the question |
| Yellow | 0.50--0.79 | Output partially addresses the question |
| Red | < 0.50 | Output does not address the question |

---

## context_precision -- Context precision (requires ground truth)

**What it measures:** Precision of retrieved contexts relative to ground truth -- how much of what the retriever pulled in was actually relevant.

**What it tells you:** Whether the retriever is surfacing relevant content or flooding the context window with noise.

**What action it drives:** Low precision means the agent wastes tokens on irrelevant context. Review your retrieval pipeline's chunking and ranking.

**How it's computed:** Ragas ContextPrecisionWithReference metric. Requires ground truth entries matched to sessions via `ground_truth.path` in the manifest.

**N/A conditions:** Scores 0.0 with `details.skipped: "no ground truth"` when no sessions have matched ground truth entries. Run `raki validate` to check your ground truth match rate.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Retrieved context is mostly relevant |
| Yellow | 0.50--0.79 | Significant irrelevant context |
| Red | < 0.50 | More noise than signal |

---

## context_recall -- Context recall (requires ground truth)

**What it measures:** Recall of retrieved contexts relative to ground truth -- how much of the needed information the retriever successfully found.

**What it tells you:** Whether the retriever is finding all the relevant knowledge.

**What action it drives:** Low recall means the agent cannot find what it needs. Check that your knowledge base contains the expected content and that search is surfacing it.

**How it's computed:** Ragas ContextRecallWithReference metric. Requires ground truth entries matched to sessions.

**N/A conditions:** Same as context_precision -- requires matched ground truth.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Most relevant knowledge is retrieved |
| Yellow | 0.50--0.79 | Important context is being missed |
| Red | < 0.50 | Retrieval fails to surface relevant knowledge |

## Ground truth requirements

Context precision and context recall require ground truth data configured in your manifest:

```yaml
ground_truth:
  path: ./ground-truth/
```

Ground truth entries are matched to sessions by `code_area` domain-token overlap from triage phases. Sessions without a triage phase will not match. Run `raki validate` to check your match rate.

See [Ground Truth Curation Guide](../curation-guide.md) for writing effective ground truth entries.
