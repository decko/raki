# Results Interpretation Reference

How to read RAKI metrics and what to do when they signal problems.
See [Getting Started](getting-started.md) for running your first evaluation.

For detailed metric documentation, see the per-tier references:
- [Operational Metrics](metrics/operational.md)
- [Knowledge Metrics](metrics/knowledge.md)
- [Analytical Metrics](metrics/analytical.md)

## Operational Health Metrics

These metrics require no LLM -- they are computed directly from session data.

### first_pass_success_rate -- First-pass success rate

Fraction of sessions that completed without any rework cycles (`session.rework_cycles == 0`).
Consistent with the `rework_cycles` metric — both read from the same `SessionMeta` field.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Most sessions pass on first try |
| Yellow | 0.60 -- 0.84 | Noticeable first-attempt failures |
| Red | < 0.60 | Majority of sessions need rework before passing |

**Red zone action:** Examine failing sessions for common root causes. Low first-pass rates often indicate unclear requirements or missing constraints in the knowledge base.

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

### self_correction_rate -- Self-correction rate

Ratio of rework findings (critical/major) resolved by the agent (higher is better). Returns N/A when no rework findings exist.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Agent fixes most issues after feedback |
| Yellow | 0.50 -- 0.79 | Some issues persist after rework |
| Red | < 0.50 | Agent fails to resolve most issues |

**Red zone action:** Investigate whether review criteria are clear and whether the agent's fix-apply loop is working. Low self-correction with high rework cycles means the agent is churning without converging.

### phase_execution_time -- Phase execution time

Mean total phase execution time per session in seconds (lower is better). Sums `duration_ms` across all phases in each session and converts to seconds. Does not capture inter-phase gaps, orchestration overhead, or human-in-the-loop pauses.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 60s | Phases complete quickly |
| Yellow | 60s -- 300s | Noticeable execution time |
| Red | > 300s | Phases are taking a long time to execute |

**Red zone action:** Examine which phases are slowest. Long execution times usually come from expensive tool calls, large context processing, or excessive retry loops within a single phase.

### token_efficiency -- Tokens / phase

Average tokens (input + output) consumed per phase (lower is better). Measures how efficiently each phase uses context. Phases where both `tokens_in` and `tokens_out` are missing are skipped.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 2,000 | Phases are lean and efficient |
| Yellow | 2,000 -- 8,000 | Moderate token usage per phase |
| Red | > 8,000 | Phases consume excessive tokens |

**Red zone action:** Check whether phases are receiving overly large context windows or generating verbose output. Reducing retrieved context size and tightening prompts are the most effective levers. High token counts combined with high cost_per_session confirm that token volume is driving spend.

## Knowledge Metrics

These metrics require `--docs-path` or `docs.path` in the manifest. They measure how well your documentation covers the domains where the agent makes mistakes.

### knowledge_gap_rate -- Knowledge gap rate

Ratio of rework findings in domains NOT covered by the knowledge base (lower is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.20 | KB covers most failure domains |
| Yellow | 0.20 -- 0.40 | Notable gaps in knowledge coverage |
| Red | > 0.40 | KB is missing content for many failure modes |

**Red zone action:** Extract the topics from uncovered findings and add them to your knowledge base. This is the single most direct way to improve agent quality.

### knowledge_miss_rate -- Knowledge miss rate

Ratio of rework findings in domains covered by the KB but the agent still got wrong (lower is better).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.10 | Agent uses KB content effectively |
| Yellow | 0.10 -- 0.30 | Agent sometimes ignores available knowledge |
| Red | > 0.30 | Agent frequently fails despite having KB coverage |

**Red zone action:** Review the KB content for affected domains. The information exists but is not preventing mistakes -- it may need restructuring or clearer language.

## Analytical Metrics

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

**Low first_pass_success_rate + high rework_cycles:** Systemic quality issues. The agent consistently requires rework, and multiple rework rounds don't fully resolve problems. Investigate whether review criteria are clear and whether the knowledge base covers the relevant domains.

**High cost_per_session + low rework_cycles:** Sessions are expensive but correct on the first pass. The cost likely comes from large context windows or verbose tool usage rather than iteration. Optimize context size or retrieval precision to reduce token consumption.

**High faithfulness + low answer_relevancy:** The agent's output is well-grounded in retrieved context but does not address the actual question. This points to a retrieval or routing problem -- the right documents are not being matched to the right questions, so the agent faithfully answers the wrong question.
