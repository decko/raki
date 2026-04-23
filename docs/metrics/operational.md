# Operational Metrics Reference

Operational metrics run with zero configuration -- no LLM, no API keys, no docs path. They are computed directly from session transcript data.

## first_pass_success_rate -- First-pass success rate

**What it measures:** Fraction of sessions that completed without any rework cycles.

**What it tells you:** Whether the agent consistently delivers correct work on the first attempt, consistent with the `rework_cycles` metric. A proxy for implementation quality.

**What action it drives:** Low values mean the implement phase is producing defective output. Investigate common failure patterns in failing sessions and improve prompts or knowledge coverage for those domains.

**How it's computed:** Count sessions where `session.rework_cycles == 0`, divided by total sessions. Returns a ratio (0.0--1.0). Returns N/A for empty datasets.

**N/A conditions:** Returns `score=None` (displayed as N/A) when the dataset is empty.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Most sessions pass on first try |
| Yellow | 0.60--0.84 | Noticeable first-attempt failures |
| Red | < 0.60 | Majority of sessions need rework |

---

## rework_cycles -- Rework cycles

**What it measures:** Mean number of review-fix iterations per session.

**What it tells you:** How much back-and-forth happens between the agent and reviewers. High values mean inefficient iteration.

**What action it drives:** Audit review findings across sessions. Repeated rework on the same issue category means the agent lacks the knowledge to get it right the first time. Add that knowledge to the KB or improve prompts.

**How it's computed:** Sum `session.rework_cycles` across all sessions, divide by session count. Returns a raw count (not 0--1 normalized).

**N/A conditions:** Never N/A -- returns 0.0 when there are no sessions.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 1.5 | Minor or no rework needed |
| Yellow | 1.5--3.0 | Sessions routinely need multiple iterations |
| Red | > 3.0 | Excessive iteration |

---

## review_severity_distribution -- Severity score

**What it measures:** Weighted severity of review findings across all sessions, where 1.0 means no findings.

**What it tells you:** Whether review findings are trending toward critical or staying minor.

**What action it drives:** Categorize findings by severity and domain. Critical findings in a specific domain point to knowledge gaps in that area.

**How it's computed:**
```
weighted = (3 * critical + 2 * major + 1 * minor) / (3 * total)
score = 1.0 - weighted
```
Returns 1.0 when there are zero findings.

**N/A conditions:** Never N/A -- returns 1.0 when no findings exist.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Few or no significant findings |
| Yellow | 0.60--0.84 | Moderate findings present |
| Red | < 0.60 | Critical or major findings are common |

---

## cost_efficiency -- Cost / session

**What it measures:** Mean LLM API cost per session in USD.

**What it tells you:** How much each agent task costs. Useful for budgeting and spotting cost spikes.

**What action it drives:** High cost usually stems from excessive rework or large context windows. Reducing retrieved context size or improving first-pass quality brings cost down.

**How it's computed:** Average `session.total_cost_usd` across sessions that have cost data. Returns raw USD value.

**N/A conditions:** Returns 0.0 when no sessions have `total_cost_usd`. The report shows "N/A" when `details.sessions_with_cost` is 0.

---

## self_correction_rate -- Self-correction rate

**What it measures:** Ratio of rework findings (critical/major) that were resolved by the agent.

**What it tells you:** Whether the agent can fix its own mistakes after review feedback.

**What action it drives:** Low self-correction with high rework cycles means the agent is churning without converging. Investigate whether review criteria are clear and whether the agent's fix-apply loop is working.

**How it's computed:** For sessions with `rework_cycles > 0` and critical/major findings: if the final verify phase has `status=completed`, all findings in that session count as resolved. Score = `resolved_findings / total_rework_findings`.

**N/A conditions:** Returns `score=None` (displayed as N/A) when no sessions have rework findings. This is normal for high-quality runs.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Agent fixes most issues after feedback |
| Yellow | 0.50--0.79 | Some issues persist after rework |
| Red | < 0.50 | Agent fails to resolve most issues |

---

## phase_execution_time -- Phase execution time

**What it measures:** Mean total phase execution time per session in seconds.

**What it tells you:** How long the agent spends on each session. Sums `duration_ms` across all phases and converts to seconds. Does not capture inter-phase gaps or human-in-the-loop pauses.

**What action it drives:** Long execution times usually come from expensive tool calls, large context processing, or retry loops within a single phase. Check which phases are slowest.

**How it's computed:** For each session with duration data, sum `phase.duration_ms` across phases, convert to seconds. Average across sessions.

**N/A conditions:** Returns 0.0 when no sessions have `duration_ms` data. The report shows "N/A" when `details.sessions_with_duration` is 0.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 60s | Phases complete quickly |
| Yellow | 60s--300s | Noticeable execution time |
| Red | > 300s | Phases take a long time |

---

## token_efficiency -- Tokens / phase

**What it measures:** Average tokens (input + output) consumed per phase.

**What it tells you:** How efficiently each phase uses context. High token counts drive both cost and latency.

**What action it drives:** Reduce retrieved context size and tighten prompts. High token counts combined with high cost confirm that token volume is driving spend.

**How it's computed:** For each session, compute the mean of `(tokens_in + tokens_out)` across phases with token data. Then average across sessions.

**N/A conditions:** Returns 0.0 when no phases have token data. The report shows "N/A" when `details.sessions_with_tokens` is 0.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 2,000 | Lean and efficient |
| Yellow | 2,000--8,000 | Moderate token usage |
| Red | > 8,000 | Excessive tokens per phase |
