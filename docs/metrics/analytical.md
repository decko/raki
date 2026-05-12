# Analytical Metrics Reference

Analytical metrics use an LLM judge (via Ragas) to evaluate retrieval quality. They require the `--judge` flag and LLM provider credentials.

> **See also:** [Rationale and Interpretation Guide](rationale-and-interpretation.md) — detailed design rationale, interpretation tables, and combined metric patterns.

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
- `vertex-anthropic` (default) — Claude via Vertex AI
- `anthropic` — direct Anthropic API (requires `ANTHROPIC_API_KEY`)
- `google` — Google AI (⚠ see [silent-zero bug](#known-issue-google-provider-silent-zero-bug) below)
- `litellm` — any model via [LiteLLM](https://docs.litellm.ai/) (requires `raki[litellm]` and the appropriate provider credentials, e.g. `OPENAI_API_KEY`)

The default judge model is `claude-sonnet-4-6`. Override with `--judge-model`.

### Manifest configuration

Judge settings can also be persisted in your manifest (shipped in v0.12.0):

```yaml
judge:
  provider: vertex-anthropic
  model: claude-sonnet-4-6
```

**Priority order** (highest wins): CLI flags > manifest `judge.*` > environment variables (`RAKI_JUDGE_PROVIDER`, `RAKI_JUDGE_MODEL`) > built-in defaults.

### Embeddings

`answer_relevancy` requires embeddings in addition to the LLM judge. The embedding provider depends on your `--judge-provider`:

| Judge provider | Embedding source | Required env vars |
|---|---|---|
| `vertex-anthropic` | Google `text-embedding-004` via Vertex AI | `GOOGLE_CLOUD_PROJECT` or `VERTEXAI_PROJECT` |
| `anthropic` | Google `text-embedding-004` via Vertex AI | `GOOGLE_CLOUD_PROJECT` or `VERTEXAI_PROJECT` |
| `google` | Google `text-embedding-004` via Vertex AI | `GOOGLE_CLOUD_PROJECT` or `VERTEXAI_PROJECT` |
| `litellm` | `text-embedding-3-small` (OpenAI) via LiteLLM | `OPENAI_API_KEY` |

If the required env vars are not set, `answer_relevancy` will fail with an error.

### Installation

Install with all extras to get Ragas dependencies:

```bash
uv sync --python 3.12 --all-extras
```

Or install only what you need:

```bash
uv pip install raki[ragas,litellm]
```

> **Note:** The `ragas` extra pulls `scikit-network`, which requires a C++ compiler (`g++`).

---

## How context synthesis works

Ragas metrics require three inputs per session: `user_input` (the question), `retrieved_contexts` (what the agent had available), and `response` (what the agent produced). RAKI synthesizes these from session phase data:

### Field mapping

| Ragas field | Source | Details |
|---|---|---|
| `user_input` | 1. Ground truth question, 2. Triage approach/summary, 3. Ticket ID | First available value is used |
| `retrieved_contexts` | `implement.knowledge_context` | Split on `\n---\n` delimiters, each chunk truncated to 1000 chars, max 10 chunks |
| `response` | Structured phase data or raw implement output | Prefers triage approach + plan tasks + implement commits over raw output; truncated to 2000 chars |
| `reference` | 1. Ground truth `reference_answer`, 2. Top-K doc chunks by keyword overlap | Only used by `context_precision` and `context_recall` |

### Sessions that get skipped

A session is **excluded from Ragas scoring** when:
- No `implement` or `session` phase exists
- The implement phase has no `knowledge_context` field (i.e., no retrieval context was injected)
- After splitting `knowledge_context` on `\n---\n`, all chunks are empty

This means you may have 50 sessions but only 10 get scored if the other 40 lack `knowledge_context`. This is common with Alcove pipeline sessions and SODA sessions where the implement phase didn't receive knowledge injection. The report shows `samples_scored` and `samples_skipped` counts in each metric's details.

### Content truncation

To prevent max_tokens errors during Ragas scoring, RAKI truncates content before sending it to the LLM judge:

| Field | Max length | Notes |
|---|---|---|
| Each retrieved context chunk | 1,000 chars | Truncated at word boundary with `[truncated]` marker |
| Response | 2,000 chars | Truncated at word boundary with `[truncated]` marker |
| Max context chunks | 10 | Additional chunks are dropped |
| Each reference doc chunk | 1,000 chars | For context_precision/recall only |
| Max reference chunks | 10 | Selected by keyword overlap with user_input |

**Impact on scores:** Analytical metric scores reflect the *truncated* content, not the full context the agent had available. A faithfulness metric cannot verify claims against context that was truncated away. If your sessions have very large `knowledge_context` values (common with verbose retrieval systems), scores may be lower than expected because the judge only sees the first portion of each chunk.

The `[truncated]` marker appears in Ragas judge output and JSON reports, making it visible when truncation occurred.

### Reference chunk selection

For `context_precision` and `context_recall`, when no ground truth `reference_answer` exists, RAKI selects reference doc chunks from `--docs-path` using **keyword overlap** (bag-of-words matching) with the `user_input`. This is a simple word-overlap approach — no embeddings or semantic similarity. Chunks with more shared words rank higher. Only chunks with at least 1 shared word are included.

This selection quality directly affects precision/recall scores. If the keyword overlap selects irrelevant chunks, precision scores will appear low even if the retriever is performing well.

---

## Known issue: Google provider silent-zero bug

When using `--judge-provider google`, the [instructor#1658](https://github.com/instructor-ai/instructor/issues/1658) bug can cause Ragas to silently return `score=0.0` for all sessions. This happens when `instructor` fails to parse Google's structured output and returns a Pydantic model with default field values (`value=0.0`, `reason=None`) instead of raising a validation error.

**Symptoms:**
- All analytical metrics return exactly 0.0
- No error messages in output
- `details.silent_zero_sessions` count in the JSON report

**How RAKI handles it:** The `is_instructor_silent_zero()` detector checks for the combination of Google provider + structured result + `value=0.0` + empty `reason`. Affected sessions are **excluded from the metric average** and logged as warnings. If all sessions are affected, the metric returns `score=None` (N/A) with `skipped: "instructor#1658: Google provider returned only silent 0.0 scores"`.

**Recommendation:** Use `--judge-provider vertex-anthropic` (the default) to avoid this issue entirely.

---

## Known issue: max_tokens failures

When session contexts are very large, the LLM judge may hit its output token limit during scoring. This produces a `max_tokens` error for that session.

**How RAKI handles it:** Affected sessions are excluded from the metric average. If all sessions fail with max_tokens errors, the metric returns `score=None` (N/A) with `skipped: "max_tokens: all sessions exceeded output token limit"`. The `details.max_tokens_sessions` count shows how many sessions were affected.

**Mitigation:** RAKI already truncates contexts to 1000 chars each (see [content truncation](#content-truncation)). If you still see max_tokens failures, your sessions may have exceptionally large numbers of context chunks (capped at 10) or very long `user_input` fields. Consider reducing the context injected into your agent's implement phase.

---

## faithfulness — Faithfulness (experimental)

**What it measures:** Whether the agent's output is grounded in the retrieved context — i.e., the agent is not hallucinating beyond what the sources provide.

**What it tells you:** How closely the agent sticks to facts in its source material.

**What action it drives:** Inspect low-scoring sessions manually. Distinguish between genuine hallucination and legitimate multi-step reasoning before taking action.

**How it's computed:** Ragas Faithfulness metric. Uses LLM-as-judge to decompose the response into individual claims and check each claim against the retrieved contexts. Each claim is scored as supported or unsupported. The score is the ratio of supported claims.

**N/A conditions:** Returns `score=None` when:
- No sessions have retrieval context (`knowledge_context` absent), OR
- All sessions failed with max_tokens errors, OR
- All sessions hit the instructor#1658 silent-zero bug (Google provider)

**Experimental caveat:** This metric was designed for natural language answers, not code. Scores may be noisy for agentic code-generation sessions where the agent synthesizes across tool calls. The details dict includes `"experimental": True` to flag this.

When the context was synthesized (inferred from phase data rather than explicit retrieval logs), the details dict includes `"context_source": "synthesized"` and metric names may appear with an `(inferred)` suffix in the report.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Output is well-grounded in context |
| Yellow | 0.50–0.79 | Some claims lack context support |
| Red | < 0.50 | Output frequently diverges from context |

---

## answer_relevancy — Answer relevancy (experimental)

**What it measures:** How relevant the agent's output is to the original user query.

**What it tells you:** Whether the agent is addressing the actual question or going off-track.

**What action it drives:** Low relevancy often indicates a prompt or routing issue rather than a retrieval problem. Check whether the agent is interpreting the task correctly.

**How it's computed:** Ragas AnswerRelevancy metric. Uses LLM to generate synthetic questions from the response, then compares them to the original `user_input` using embedding similarity. Requires both an LLM judge and an embedding model (see [embeddings](#embeddings)).

**N/A conditions:** Returns `score=None` when:
- No sessions have retrieval context, OR
- All sessions failed with max_tokens or silent-zero errors

**Experimental caveat:** Multi-step agent workflows may address the question indirectly, lowering scores without indicating a real problem. The details dict includes `"experimental": True`.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Output directly addresses the question |
| Yellow | 0.50–0.79 | Output partially addresses the question |
| Red | < 0.50 | Output does not address the question |

---

## context_precision — Context precision (requires ground truth)

**What it measures:** Precision of retrieved contexts relative to ground truth — how much of what the retriever pulled in was actually relevant.

**What it tells you:** Whether the retriever is surfacing relevant content or flooding the context window with noise.

**What action it drives:** Low precision means the agent wastes tokens on irrelevant context. Review your retrieval pipeline's chunking and ranking.

**How it's computed:** Ragas ContextPrecisionWithReference metric. Uses LLM-as-judge to evaluate each retrieved context chunk against the reference answer. Requires ground truth entries matched to sessions via `ground_truth.path` in the manifest.

When no ground truth `reference_answer` is available but `--docs-path` is provided, RAKI uses [keyword-selected doc chunks](#reference-chunk-selection) as the reference instead.

**N/A conditions:** Returns `score=None` with `details.skipped: "no ground truth"` when no sessions have matched ground truth entries or doc chunks. Run `raki validate` to check your ground truth match rate.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Retrieved context is mostly relevant |
| Yellow | 0.50–0.79 | Significant irrelevant context |
| Red | < 0.50 | More noise than signal |

---

## context_recall — Context recall (requires ground truth)

**What it measures:** Recall of retrieved contexts relative to ground truth — how much of the needed information the retriever successfully found.

**What it tells you:** Whether the retriever is finding all the relevant knowledge.

**What action it drives:** Low recall means the agent cannot find what it needs. Check that your knowledge base contains the expected content and that search is surfacing it.

**How it's computed:** Ragas ContextRecall metric. Uses LLM-as-judge to check whether each piece of information in the reference answer can be found in the retrieved contexts. Requires ground truth entries matched to sessions.

**N/A conditions:** Same as context_precision — returns `score=None` when no sessions have matched ground truth or doc chunks.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Most relevant knowledge is retrieved |
| Yellow | 0.50–0.79 | Important context is being missed |
| Red | < 0.50 | Retrieval fails to surface relevant knowledge |

---

## Ground truth requirements

Context precision and context recall require ground truth data configured in your manifest:

```yaml
ground_truth:
  path: ./ground-truth/
```

Ground truth entries are matched to sessions by `code_area` domain-token overlap from triage phases. Sessions without a triage phase will not match. Run `raki validate` to check your match rate.

See [Ground Truth Curation Guide](../curation-guide.md) for writing effective ground truth entries.

---

## Debugging analytical metrics

**All metrics return N/A despite providing `--judge`:**
- Check that sessions have `knowledge_context` in their implement phase — sessions without it are [skipped](#sessions-that-get-skipped)
- Run `raki validate --deep` to verify adapter and metric health

**Scores are all exactly 0.0:**
- If using `--judge-provider google`, this is likely the [silent-zero bug](#known-issue-google-provider-silent-zero-bug)
- Switch to `--judge-provider vertex-anthropic`
- Check `details.silent_zero_sessions` in the JSON report

**Scores are unexpectedly low:**
- Check if [content truncation](#content-truncation) is cutting off important context — look for `[truncated]` markers in the JSON report
- For `answer_relevancy`, verify the `user_input` field is meaningful — if it falls back to ticket ID, relevancy scores will be low
- For `context_precision`/`context_recall`, check whether the reference was keyword-selected doc chunks (may not be semantically relevant)

**Some sessions scored, others skipped:**
- Check `details.samples_scored` vs `details.samples_skipped` in the JSON report
- Skipped sessions typically lack `knowledge_context` or had scoring errors
- `details.max_tokens_sessions` shows how many failed from context size
