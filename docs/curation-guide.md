# Ground Truth Curation Guide

Ground truth entries define what correct retrieval looks like for a given question. RAKI compares what an agent actually retrieved against these entries to compute retrieval quality metrics. Without well-crafted ground truth, metrics like context recall and precision are meaningless -- they measure against garbage in, garbage out.

## Writing Good Questions

A good question is **discriminating**: it can only be answered correctly if the right knowledge was retrieved. Aim for questions that test whether the retrieval system found the relevant context, not whether an LLM can guess the answer from general knowledge.

- Ask about project-specific decisions, constraints, or procedures
- Require information that lives in a specific part of the knowledge base
- Avoid questions answerable from the codebase structure alone (e.g., "What language is this written in?")
- Avoid questions so broad they match everything (e.g., "How does the system work?")

## Writing `expected_contexts`

Each entry in `expected_contexts` is a semantic description of what should have been retrieved. RAKI matches these against actual retrieved contexts using semantic similarity, not exact string matching.

- **Do**: Write concise descriptions of the key information: `"JWT bearer token validation on all protected routes"`
- **Don't**: Copy-paste verbatim sentences from source documents
- **Don't**: Write vague descriptions that match too broadly: `"something about authentication"`
- Include 1-3 entries per question. Each should capture a distinct piece of required knowledge.

## Difficulty Calibration

| Level    | Description                              | Example                                          |
|----------|------------------------------------------|--------------------------------------------------|
| `easy`   | Single fact lookup from one source        | "What validation framework does the project use?" |
| `medium` | Multi-step reasoning or multiple sources  | "What middleware is required for API endpoints?"   |
| `hard`   | Cross-domain synthesis of 3+ sources      | "How do retry, auth, and rate limiting interact?"  |

## Knowledge Type Selection

| Type                 | Use when the answer is...                                    |
|----------------------|--------------------------------------------------------------|
| `fact`               | A declarative statement ("the project uses Pydantic")        |
| `procedure`          | A sequence of steps or a how-to                              |
| `constraint`         | A rule, limit, or invariant ("100 req/min per API key")      |
| `context-dependent`  | The answer depends on combining multiple domain contexts      |

## Domain Assignment

Use a controlled vocabulary for domains -- reuse existing terms rather than inventing synonyms. One entry can belong to multiple domains when it spans concerns. Check existing entries in your ground truth file before adding new domain labels.

```yaml
# Good: reuses existing domain terms
domains: [api, auth, middleware]

# Bad: synonyms that fragment the vocabulary
domains: [authentication]     # use "auth" instead
domains: [rest-api]           # use "api" instead
```

## Anti-Patterns

Avoid these common mistakes:

- **Too broad** -- `"How does the system work?"` has no specific retrieval target
- **Too obvious** -- `"What programming language is the code written in?"` is answerable from file extensions
- **Verbatim contexts** -- copying full sentences from source documents defeats semantic matching; write concise descriptions instead
- **Duplicate coverage** -- three entries testing the same concept waste effort and skew metrics

## Worked Examples

### Good Entry

```yaml
- question: "What are the rate limiting constraints for the public API?"
  expected_contexts:
    - "100 requests per minute per API key"
    - "429 Too Many Requests with Retry-After header"
  domains: [api, middleware, security]
  difficulty: easy
  knowledge_type: constraint
  expected_phase: triage
```

This works because the question targets a specific constraint, the contexts are concise semantic descriptions, and difficulty is calibrated correctly (single fact lookup).

### Bad Entry (and why)

```yaml
- question: "Tell me about the API"
  expected_contexts:
    - "The API has rate limiting and authentication and validation"
  domains: [general]
  difficulty: easy
  knowledge_type: fact
```

Problems: the question is too vague to discriminate retrieval quality, the context is a grab-bag that matches too broadly, `"general"` is not a useful domain, and difficulty should be `hard` if it genuinely requires synthesizing multiple topics.

### Good Entry (hard difficulty)

```yaml
- question: >
    When a WebSocket notification fails to deliver, how should the system
    behave given the rate limiting and auth constraints?
  expected_contexts:
    - "retry with exponential backoff up to 3 attempts"
    - "dead-letter queue for undeliverable notifications"
    - "tenant auth token refresh before retry"
  domains: [api, notifications, auth, middleware]
  difficulty: hard
  knowledge_type: context-dependent
  expected_phase: implement
```

This works because it requires synthesizing knowledge across notifications, auth, and rate limiting -- three separate domains. The contexts each capture a distinct required piece.

## How Matching Works

RAKI matches ground truth entries to sessions using domain-token overlap. During evaluation, each session's triage phase `output_structured["code_area"]` field is split into domain tokens. These tokens are compared against the `domains` list in each ground truth entry. The entry with the highest overlap is assigned to the session.

**Important:** Sessions without a triage phase or without a `code_area` field in their structured output will not match any ground truth entry. If you see a low match rate (below 50%), check that your sessions include triage phases with `code_area` populated.

You can verify match rates before running a full evaluation:

```bash
raki validate -m raki.yaml
```

The validate command shows a ground truth match preview when `ground_truth.path` is configured in your manifest.

## Further Reading

- [`examples/ground-truth/curated.yaml`](../examples/ground-truth/curated.yaml) -- five fully annotated entries covering all difficulty levels and knowledge types
- [`docs/interpretation-reference.md`](interpretation-reference.md) -- what the resulting metrics mean and how to act on them
