# Results Interpretation Reference

How to read RAKI metrics and what to do when they signal problems.
See [Getting Started](getting-started.md) for running your first evaluation.

## Operational Health Metrics

These metrics require no LLM -- they are computed directly from session data.

### first_pass_verify_rate -- Verify rate

Fraction of sessions that pass verification on the first attempt.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Most sessions pass on first try |
| Yellow | 0.60 -- 0.84 | Noticeable first-attempt failures |
| Red | < 0.60 | Majority of sessions need rework before passing |

**Red zone action:** Examine failing sessions for common root causes. Low verify rates often indicate unclear requirements or missing constraints in the knowledge base.

### rework_cycles -- Rework cycles

Mean number of review-rework iterations per session (lower is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 1.5 | Minor or no rework needed |
| Yellow | 1.5 -- 3.0 | Sessions routinely need multiple iterations |
| Red | > 3.0 | Excessive iteration -- review criteria may be unclear |

**Red zone action:** Audit review findings across sessions. Repeated rework on the same issue category means the agent lacks the knowledge to get it right the first time.

### review_severity_distribution -- Severity score

Weighted severity of review findings, where 1.0 means no findings at all (higher is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Few or no significant findings |
| Yellow | 0.60 -- 0.84 | Moderate findings present |
| Red | < 0.60 | Critical or major findings are common |

**Red zone action:** Categorize findings by severity and domain. Critical findings in a specific domain point to knowledge gaps in that area.

### cost_efficiency -- Cost / session

Mean USD cost per session (lower is better). Acceptable ranges depend on your context.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | Within budget | Cost is acceptable for your use case |
| Yellow | 1.5x budget | Spending more than expected |
| Red | > 2x budget | Sessions are significantly over budget |

**Red zone action:** Check token counts and rework cycles. High cost usually stems from excessive rework or overly large context windows. Reducing retrieved context or improving first-pass quality brings cost down.

### knowledge_retrieval_miss_rate -- Knowledge miss rate

Fraction of rework-triggering findings where the relevant knowledge was not in the retrieved context (lower is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.15 | Knowledge base covers most failure cases |
| Yellow | 0.15 -- 0.40 | Notable gaps in knowledge coverage |
| Red | > 0.40 | Knowledge base is missing content for many failure modes |

**Red zone action:** Extract the topics from missed findings and add them to your knowledge base. This is the single most direct way to improve agent quality.

## Retrieval Quality Metrics

These metrics use LLM-backed evaluation via Ragas and require ground truth data.

> **Note:** Retrieval quality metrics require ground truth to be configured and matched. If `ground_truth.path` is not set in your manifest, or if sessions do not match any ground truth entries, these metrics will produce 0.0 scores. Run `raki validate` to check your ground truth match rate.

### context_precision -- Context precision

Fraction of retrieved contexts that are actually relevant to the question (higher is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Retrieved context is mostly relevant |
| Yellow | 0.50 -- 0.79 | Significant irrelevant context retrieved |
| Red | < 0.50 | More noise than signal in retrieved context |

**Red zone action:** Review your retrieval pipeline's chunking and ranking. Low precision means the agent wastes tokens on irrelevant context.

### context_recall -- Context recall

Fraction of expected contexts that were actually retrieved (higher is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Most relevant knowledge is being retrieved |
| Yellow | 0.50 -- 0.79 | Important context is being missed |
| Red | < 0.50 | Retrieval is failing to surface relevant knowledge |

**Red zone action:** Check that your knowledge base contains the expected content and that embedding/search is surfacing it. Low recall means the agent cannot find what it needs.

### faithfulness -- Faithfulness (experimental)

Whether the agent's output is grounded in retrieved context (higher is better). **This metric is experimental for agentic sessions** -- agents may synthesize across tool calls in ways that lower faithfulness scores without indicating a real problem.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Output is well-grounded in context |
| Yellow | 0.50 -- 0.79 | Some claims lack context support |
| Red | < 0.50 | Output frequently diverges from context |

**Red zone action:** Inspect low-scoring sessions manually. Distinguish between genuine hallucination and legitimate multi-step reasoning before taking action.

### answer_relevancy -- Answer relevancy (experimental)

Whether the agent's output addresses the original question (higher is better). **This metric is experimental for agentic sessions** -- multi-step agent workflows may address the question indirectly.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Output directly addresses the question |
| Yellow | 0.50 -- 0.79 | Output partially addresses the question |
| Red | < 0.50 | Output does not address the question |

**Red zone action:** Check whether the agent is interpreting the task correctly. Low relevancy often indicates a prompt or routing issue rather than a retrieval problem.

## Common Patterns

When individual metrics don't tell the full story, look at combinations.

**High knowledge_miss_rate + high severity_score:** Retrieval gaps exist but happen to land on easy questions where the agent can still produce acceptable output. Add the missing knowledge now -- these gaps will eventually cause failures on harder variants.

**Low verify_rate + high rework_cycles:** Systemic quality issues. The agent consistently produces work that fails verification, and multiple rework rounds don't fully resolve problems. Investigate whether review criteria are clear and whether the knowledge base covers the relevant domains.

**High cost_per_session + low rework_cycles:** Sessions are expensive but correct on the first pass. The cost likely comes from large context windows or verbose tool usage rather than iteration. Optimize context size or retrieval precision to reduce token consumption.

**High faithfulness + low answer_relevancy:** The agent's output is well-grounded in retrieved context but does not address the actual question. This points to a retrieval or routing problem -- the right documents are not being matched to the right questions, so the agent faithfully answers the wrong question.
