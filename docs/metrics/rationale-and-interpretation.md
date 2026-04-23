# Rationale and Interpretation Guide for Non-Ragas Metrics

This guide explains **why each non-Ragas metric exists**, **what it actually measures**,
**how to interpret each value**, and **what action to take** when a metric signals a problem.

Non-Ragas metrics compute directly from session transcript data with no LLM required.
They are split into two tiers:

- **Operational metrics** — always available, measure agent efficiency and quality
- **Knowledge metrics** — require `--docs-path`, measure knowledge base coverage and effectiveness

For the complete zone tables and quick-reference thresholds, see:
- [Operational Metrics Reference](operational.md)
- [Knowledge Metrics Reference](knowledge.md)

For a high-level summary of all metrics, see [Interpreting Results](../interpreting-results.md).

---

## Why Non-Ragas Metrics?

Ragas metrics (context precision, context recall, faithfulness, answer relevancy) require an
LLM judge and ground truth data. They are powerful but expensive and unavailable in every run.

Non-Ragas metrics are always available because they derive from the session transcript
structure: which phases ran, how many rework cycles occurred, what the review findings said,
how much the session cost. They answer the **operational** question — *is the agent working
well right now?* — without any external dependencies.

These metrics are designed to be:

- **Deterministic**: the same transcript always produces the same score
- **Actionable**: each metric points to a specific intervention
- **Composable**: reading metrics in combination reveals patterns that individual metrics miss

---

## Operational Metrics

### first_pass_verify_rate

**Rationale**: The first-pass verify rate is the primary signal of implementation quality.
An agent that consistently delivers correct work on the first attempt is more reliable and
less expensive than one requiring multiple review cycles. This metric focuses on `generation=1`
because subsequent verify phases represent rework, which is already captured by `rework_cycles`.

**What it measures**: Fraction of sessions where the `verify` phase at generation 1 has
`status=completed`. A session is counted as a first-pass success only when the generation-1
verify phase passes; a failed generation-1 verify scores 0.0 for that session even if a
later generation eventually passes.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | ≥ 0.85 | Most sessions pass on first try — agent is reliable |
| Yellow | 0.60–0.84 | Noticeable first-attempt failures — investigate patterns |
| Red | < 0.60 | Majority of sessions need rework — systemic quality issue |

**Action when red**: Examine the review findings on failing sessions. Common root causes:
unclear requirements passed to the agent, missing constraints in the knowledge base, or
the agent using an outdated implementation strategy. Look for whether failures cluster
around specific task types, reviewers, or time periods.

**Pitfall**: A verify rate of 0.0 may mean no sessions have `verify` phases at all (the
report will show "N/A" in that case). Check `details.total` to distinguish zero passes
from zero opportunities.

---

### rework_cycles

**Rationale**: Rework cycles measure the cost of iteration. Each cycle represents a
review-fix loop where the agent consumed additional tokens, introduced latency, and
potentially added new defects. A session with `rework_cycles=3` is roughly 3× more
expensive in LLM calls and wall-clock time than a first-pass success. This metric uses
the raw count rather than a normalized 0–1 score because the business impact scales
linearly with cycles.

Unlike `first_pass_verify_rate` (which is binary per session), `rework_cycles` captures
the degree of iteration: an agent averaging 0.2 cycles is significantly more efficient
than one averaging 1.2 cycles, even if both occasionally fail the first pass.

**What it measures**: Mean `session.rework_cycles` across all sessions. Returns raw count.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 1.5 | Minor or no rework — agent is efficient |
| Yellow | 1.5–3.0 | Sessions routinely need multiple iterations |
| Red | > 3.0 | Excessive iteration — high cost, likely a systemic issue |

**Action when red**: Audit review findings across sessions for repeated patterns. Repeated
rework on the same issue category means the agent lacks the knowledge to get it right the
first time. Add that knowledge to the KB or improve prompts to prevent the known failure mode.

**Combined signal**: High `rework_cycles` + low `first_pass_verify_rate` confirms a
systemic quality problem. High `rework_cycles` + high `self_correction_rate` means the
agent iterates a lot but does eventually converge — the cost is high but the outcome is good.

---

### review_severity_distribution

**Rationale**: Not all findings are equal. A critical finding (broken functionality,
security flaw) is far more damaging than a minor style nit. The weighted severity score
assigns critical findings a weight of 3, major a weight of 2, and minor a weight of 1,
then normalizes so that 1.0 represents a completely clean run.

This weighting correctly identifies runs where a handful of critical findings indicate a
systemic problem, even if the total finding count is low. The score is inverted (higher
is better) to follow the convention that improving scores move upward.

**What it measures**: `score = 1.0 - (3*critical + 2*major + 1*minor) / (3*total)`.
Returns 1.0 when no findings exist.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | ≥ 0.85 | Few or no significant findings |
| Yellow | 0.60–0.84 | Moderate findings present |
| Red | < 0.60 | Critical or major findings are common |

**Practical example**: 3 critical + 0 other findings → weighted = `(9+0+0)/9 = 1.0` →
score = `1.0 - 1.0 = 0.0`. Contrast with 0 critical + 30 minor findings →
weighted = `(0+0+30)/90 ≈ 0.33` → score ≈ 0.67. The first case correctly scores worse
despite having fewer total findings.

**Action when red**: Categorize findings by severity and domain. Critical findings in a
specific domain point to knowledge gaps in that area — cross-reference with
`knowledge_gap_rate` to see if those domains are missing from the KB.

---

### cost_efficiency

**Rationale**: LLM API cost is the most direct financial measure of agent efficiency.
Unlike the other metrics, cost does not have a universal target threshold because acceptable
cost depends on the value of the work being automated. Instead, it is tracked to identify
outliers and trends.

High cost combined with high `rework_cycles` confirms that iteration is the primary cost
driver. High cost with low `rework_cycles` points to large context windows or verbose tool
usage as the culprit. The metric reports raw USD rather than a normalized score because
cost thresholds are project-specific.

**What it measures**: Mean `session.total_cost_usd` across sessions with cost data.

**N/A conditions**: Returns 0.0 when no sessions have `total_cost_usd` logged. The report
shows "N/A" when `details.sessions_with_cost == 0`.

**How to interpret**: There are no fixed zones — establish a baseline from the first few
runs and flag sessions that cost significantly more than the median.

**Action when high**: Check `token_efficiency` and `rework_cycles` together. If tokens are
high, reduce context window size or improve retrieval precision. If rework is high, improving
first-pass quality will reduce cost proportionally.

**Combined signal**: High `cost_efficiency` + high `token_efficiency` confirms that token
volume is the primary cost driver. High `cost_efficiency` + low `token_efficiency` suggests
a pricing-tier or model difference rather than a context size problem.

---

### self_correction_rate

**Rationale**: When an agent makes a mistake, the critical question is: can it fix it?
Self-correction rate measures whether the agent can apply reviewer feedback and deliver
a correct result in subsequent generations. A high self-correction rate means the agent
is effectively learning from feedback within a session. A low rate means the agent is
churning — consuming tokens and time without converging on a correct answer.

Only critical and major findings are counted because minor findings rarely block final
verification. A session's findings are considered resolved when its final verify phase
has `status=completed`.

**What it measures**: `resolved_findings / total_rework_findings` across sessions with
rework. Returns `None` (N/A) when no rework findings exist — this is normal for
high-quality runs.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | ≥ 0.80 | Agent fixes most issues after feedback |
| Yellow | 0.50–0.79 | Some issues persist after rework |
| Red | < 0.50 | Agent fails to resolve most issues |
| N/A | — | No rework findings — expected for clean runs |

**Action when red**: Investigate whether review criteria are clear and whether the
agent's fix-apply loop is working. Low self-correction with high `rework_cycles` means
the agent is churning without converging — the review criteria may be ambiguous, or the
agent may not be receiving the finding details it needs to fix the issue.

**Pitfall**: A high self-correction rate does not mean the agent is good — it just means
it can recover. An agent that always needs 3 rework cycles but eventually passes has high
self-correction but poor `first_pass_verify_rate`.

---

### phase_execution_time

**Rationale**: Phase execution time captures the sum of `duration_ms` across all phases
in a session, converted to seconds. It measures only the time the agent actively spent
processing, excluding inter-phase gaps, orchestration overhead, and human-in-the-loop
pauses. This makes it a reliable proxy for LLM call duration rather than total wall-clock
time.

High execution time combined with high `token_efficiency` confirms that expensive LLM
calls are the latency bottleneck. High time with low tokens suggests slow tool calls or
API rate limiting rather than a model problem.

**What it measures**: Mean total phase time per session in seconds. Reports p50, p95, min,
and max in `details` to distinguish typical performance from tail latency.

**N/A conditions**: Returns 0.0 when no sessions have `duration_ms` data. The report shows
"N/A" when `details.sessions_with_duration == 0`.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 60s | Phases complete quickly |
| Yellow | 60s–300s | Noticeable execution time |
| Red | > 300s | Phases are taking a long time |

**Action when red**: Check which phases are slowest by examining per-session phase details.
High `p95` with low `p50` means a small number of sessions are outliers — investigate those
specifically. Long execution times usually come from large context processing or retry loops.

---

### token_efficiency

**Rationale**: Tokens are the fundamental unit of LLM cost and a primary driver of latency.
By computing the average of `(tokens_in + tokens_out)` per phase rather than per session,
this metric isolates context consumption at the phase level, making it comparable across
sessions with different numbers of phases.

High `tokens_in` relative to `tokens_out` suggests over-retrieval (too much context fed in).
High `tokens_out` suggests verbose generation that may not be necessary. Combined with
`cost_efficiency`, this metric confirms whether token volume is the primary cost driver.

**What it measures**: Mean of `(tokens_in + tokens_out)` per phase, averaged across
sessions. Phases where both `tokens_in` and `tokens_out` are `None` are skipped.

**N/A conditions**: Returns 0.0 when no phases have token data. The report shows "N/A"
when `details.sessions_with_tokens == 0`.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 2,000 | Phases are lean and efficient |
| Yellow | 2,000–8,000 | Moderate token usage per phase |
| Red | > 8,000 | Phases consume excessive tokens |

**Action when red**: Reduce retrieved context size by improving retrieval precision or
adjusting chunk sizes. Tighten prompts to reduce verbose generation. Check whether certain
phases consistently over-consume — implement phase-specific context limits.

**Combined signal**: High `token_efficiency` + high `cost_efficiency` confirms that token
volume is driving cost. High `token_efficiency` + low `phase_execution_time` suggests fast
but expensive model calls.

---

## Knowledge Metrics

These metrics require `--docs-path` or `docs.path` in your manifest. They measure how well
your knowledge base covers the domains where the agent makes mistakes.

### knowledge_gap_rate

**Rationale**: When an agent fails on a task, the first diagnostic question is: did it have
the right reference material? Knowledge gap rate answers this by checking whether the domains
where critical and major failures occurred are covered by the knowledge base.

A high gap rate is directly actionable: the uncovered findings point to specific topics that
need to be added to the knowledge base. Unlike Ragas context precision/recall (which measure
retrieval quality within the KB), gap rate measures coverage — whether the content exists at
all.

**What it measures**: `uncovered_findings / total_rework_findings` where "uncovered" means
the finding's issue words do not overlap with any domain's doc content. Returns `None` (N/A)
when no rework findings exist or no docs are loaded.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.20 | KB covers >80% of failure domains |
| Yellow | 0.20–0.40 | Notable gaps in knowledge coverage |
| Red | > 0.40 | KB is missing content for many failure modes |

**Action when red**: Extract the topics from uncovered findings and add them to your KB.
This is the single most direct way to improve agent quality — the agent is failing because
the reference material it needs simply does not exist.

**Pitfall**: Gap rate uses word overlap to determine coverage. Synonyms or domain-specific
terminology in findings that does not appear in doc chunks will cause false negatives
(findings appear uncovered when they are actually covered by differently-worded content).
Review uncovered findings manually before investing in new documentation.

**Relationship to miss rate**: `knowledge_gap_rate + knowledge_miss_rate` may not sum to
1.0 because minor findings are excluded and sessions without knowledge context are skipped
entirely. See [knowledge.md](knowledge.md) for the full relationship table.

---

### knowledge_miss_rate

**Rationale**: Knowledge miss rate addresses the follow-up question to gap rate: if the KB
has coverage for a domain, is the agent using it effectively? A high miss rate means the
agent is failing in domains where documentation exists — the agent may not be retrieving the
right content, may be retrieving it but ignoring it, or the content may be poorly structured
or outdated.

This is the inverse complement of gap rate: a high miss rate with low gap rate means your KB
is comprehensive but ineffective, pointing to a retrieval quality or prompt engineering
problem rather than a documentation coverage gap.

**What it measures**: `covered_findings / total_rework_findings` where "covered" means the
finding's issue words overlap with at least one domain's doc content. Returns `None` (N/A)
when no rework findings exist or no docs are loaded.

**How to interpret**:

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.10 | Agent uses available KB content effectively |
| Yellow | 0.10–0.30 | Agent sometimes ignores available knowledge |
| Red | > 0.30 | Agent frequently fails despite having KB coverage |

**Action when red**: Review the KB content for the affected domains. The information exists
but is not preventing mistakes — it may need to be restructured, made more explicit, or
moved closer to the agent's context window. Check retrieval logs to see if the relevant
chunks are actually being surfaced.

---

## Reading Metrics in Combination

Individual metrics rarely tell the full story. The most useful signals emerge from
combining multiple metrics.

### Pattern 1: Systemic Quality Issue

**Signals**: Low `first_pass_verify_rate` + High `rework_cycles`

**Interpretation**: The agent consistently produces work that fails verification, and
multiple rework rounds don't fully resolve problems. This is the worst-case combination.

**Action**: Investigate whether review criteria are clear and whether the KB covers the
relevant domains. Run with `--docs-path` to enable knowledge metrics and identify specific
coverage gaps.

---

### Pattern 2: KB Coverage Gap

**Signals**: High `knowledge_gap_rate` + High `review_severity_distribution` (red zone)

**Interpretation**: The agent is failing on tasks where reference material is missing. The
review findings point to specific domains not covered by the KB.

**Action**: Extract topics from uncovered findings. Add documentation for those domains.
After updating the KB, re-run to verify the gap rate improves.

---

### Pattern 3: KB Exists But Is Ineffective

**Signals**: Low `knowledge_gap_rate` + High `knowledge_miss_rate`

**Interpretation**: The KB covers the right domains, but the agent fails anyway. The content
exists but is not preventing mistakes.

**Action**: Review KB content for clarity and specificity. Check whether retrieval is
surfacing the right chunks. Consider restructuring documentation or improving retrieval
precision.

---

### Pattern 4: Expensive but Correct

**Signals**: High `cost_efficiency` + Low `rework_cycles` + High `first_pass_verify_rate`

**Interpretation**: Sessions are expensive but correct on the first pass. The cost likely
comes from large context windows or verbose tool usage rather than iteration.

**Action**: Optimize context size or retrieval precision to reduce token consumption. Check
`token_efficiency` to confirm that token volume is the driver.

---

### Pattern 5: Agent Converges But Slowly

**Signals**: High `rework_cycles` + High `self_correction_rate`

**Interpretation**: The agent needs multiple iterations but does eventually produce correct
work. Throughput and cost suffer but quality is acceptable.

**Action**: Focus on first-pass quality (prompts, KB coverage) to reduce the number of
cycles needed. The self-correction mechanism is working — the bottleneck is upstream.

---

### Pattern 6: Agent Churns Without Converging

**Signals**: High `rework_cycles` + Low `self_correction_rate`

**Interpretation**: The agent iterates repeatedly but fails to resolve the review findings.
This is a signal that either the review criteria are ambiguous or the agent's fix-apply
loop is broken.

**Action**: Check whether review findings contain enough detail for the agent to understand
the required fix. Verify that finding text is passed to the agent in subsequent generations.
Consider adding examples of correct fixes to the KB for common finding types.

---

## Baseline and Trend Analysis

All non-Ragas metrics support trend analysis across runs. The recommended workflow is:

1. **Establish a baseline** from the first 3–5 runs on real sessions
2. **Set informal targets** based on the baseline (e.g., "reduce rework cycles by 0.5")
3. **Use `raki report --diff`** to compare consecutive runs and detect regressions
4. **Set quality gates** via `--gate` to enforce minimum standards in CI:

   ```bash
   raki run --gate "first_pass_verify_rate>0.80" --gate "rework_cycles<2.0"
   ```

5. **Investigate outlier sessions** — sort by cost and examine the most expensive ones first

For CI integration, see [CI Integration Guide](../ci-integration.md).
