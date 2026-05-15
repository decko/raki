# Operational Metrics Reference

Operational metrics run with zero configuration — no LLM, no API keys, no docs path. They are computed directly from session transcript data.

> **Note:** The zone thresholds below (green/yellow/red) are defaults derived from real SODA pipeline session data. They are reasonable starting points but may need adjustment for your agent system and task complexity. Use `--gate` to set project-specific thresholds.

## first_pass_success_rate — First-pass success rate

**What it measures:** Fraction of sessions that completed without any rework cycles.

**What it tells you:** Whether the agent consistently delivers correct work on the first attempt, consistent with the `rework_cycles` metric. A proxy for implementation quality.

**What action it drives:** Low values mean the implement phase is producing defective output. Investigate common failure patterns in failing sessions and improve prompts or knowledge coverage for those domains.

**How it's computed:** Count sessions where `session.rework_cycles == 0`, divided by total sessions. Returns a ratio (0.0–1.0). Returns `score=None` (N/A) when the dataset is empty — a dataset with sessions that all have rework returns 0.0, not N/A.

**N/A conditions:** Returns `score=None` (displayed as N/A) when the dataset is empty (zero sessions).

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Most sessions pass on first try |
| Yellow | 0.60–0.84 | Noticeable first-attempt failures |
| Red | < 0.60 | Majority of sessions need rework |

---

## rework_cycles — Rework cycles

**What it measures:** Mean number of review-fix iterations per session.

**What it tells you:** How much back-and-forth happens between the agent and reviewers. High values mean inefficient iteration.

**What action it drives:** Audit review findings across sessions. Repeated rework on the same issue category means the agent lacks the knowledge to get it right the first time. Add that knowledge to the KB or improve prompts.

**How it's computed:** Sum `session.rework_cycles` across all sessions, divide by session count. Returns a raw count (not 0–1 normalized).

**N/A conditions:** Never returns `score=None` — returns `score=0.0` for empty datasets. This means a score of 0.0 can indicate either "no rework happened" or "no sessions exist." Check `details.sessions` to distinguish: if `sessions > 0`, it genuinely means zero rework.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 1.5 | Minor or no rework needed |
| Yellow | 1.5–3.0 | Sessions routinely need multiple iterations |
| Red | > 3.0 | Excessive iteration |

---

## review_severity_distribution — Severity score

**What it measures:** Weighted severity of review findings across all sessions, where 1.0 means no findings.

**What it tells you:** Whether review findings are trending toward critical or staying minor.

**What action it drives:** Categorize findings by severity and domain. Critical findings in a specific domain point to knowledge gaps in that area.

**How it's computed:**

```
weighted = (3 * critical + 2 * major + 1 * minor) / (3 * total)
score = 1.0 - weighted
```

The denominator `3 * total` represents the worst case: if every finding were critical. This normalization means the score measures how far the actual severity distribution is from the worst case, not from the average case.

**Pool-based aggregation:** All findings from all sessions are pooled together — a single session with 50 minor findings has more influence than a session with 1 critical finding, despite the latter being a more serious quality signal. This is by design: the metric measures the *overall* severity landscape, not per-session quality. Use `self_correction_rate` and `first_pass_success_rate` for per-session quality signals.

Returns 1.0 when there are zero findings.

**Practical example:** 3 critical + 0 other findings → `weighted = 9/9 = 1.0` → score = `0.0`. Contrast with 0 critical + 30 minor findings → `weighted = 30/90 ≈ 0.33` → score ≈ `0.67`. The first case correctly scores worse despite having fewer total findings.

**N/A conditions:** Never N/A — returns 1.0 when no findings exist.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.85 | Few or no significant findings |
| Yellow | 0.60–0.84 | Moderate findings present |
| Red | < 0.60 | Critical or major findings are common |

---

## cost_efficiency — Cost / session

**What it measures:** Mean LLM API cost per session in USD.

**What it tells you:** How much each agent task costs. Useful for budgeting and spotting cost spikes.

**What action it drives:** High cost usually stems from excessive rework or large context windows. Reducing retrieved context size or improving first-pass quality brings cost down.

**How it's computed:** Average `session.total_cost_usd` across sessions that have cost data. Returns raw USD value.

**N/A conditions:** Returns `score=0.0` internally when no sessions have `total_cost_usd`. The report detects this via `details.sessions_with_cost == 0` and displays "N/A" instead of the misleading $0.00. This is a display concern — the metric itself cannot distinguish "zero cost" from "no data" because it returns a numeric score, not `None`.

---

## self_correction_rate — Self-correction rate

**What it measures:** Ratio of rework findings (critical/major, excluding synthesized) that were resolved by the agent.

**What it tells you:** Whether the agent can fix its own mistakes after review feedback.

**What action it drives:** Low self-correction with high rework cycles means the agent is churning without converging. Investigate whether review criteria are clear and whether the agent's fix-apply loop is working.

**How it's computed:** For sessions with `rework_cycles > 0`: collect all critical/major non-synthesized findings. If the session's final verify phase (highest generation) has `status=completed`, all that session's findings count as resolved. Score = `resolved_findings / total_rework_findings`.

Synthesized findings (inferred from tool failures in the transcript) are excluded because they are not actionable review feedback — including them would artificially inflate the denominator.

**N/A conditions:** Returns `score=None` (displayed as N/A) when no sessions have rework findings. This is normal for high-quality runs where every session passes on the first attempt.

**Implementation note for contributors:** This metric uses `total_rework_findings: 0` in its N/A details dict, without the `sessions_with_` prefix that other metrics use. The CLI summary N/A detection (`_has_no_data()`) still works because it checks `score=None` separately from the `sessions_with_*` key convention. New metrics should prefer the `sessions_with_*` prefix for consistency.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Agent fixes most issues after feedback |
| Yellow | 0.50–0.79 | Some issues persist after rework |
| Red | < 0.50 | Agent fails to resolve most issues |

---

## phase_execution_time — Phase execution time

**What it measures:** Mean total phase execution time per session in seconds.

**What it tells you:** How long the agent spends on each session. Sums `duration_ms` across all phases and converts to seconds. Does not capture inter-phase gaps or human-in-the-loop pauses.

**What action it drives:** Long execution times usually come from expensive tool calls, large context processing, or retry loops within a single phase. Check which phases are slowest.

**How it's computed:** For each session with duration data, sum `phase.duration_ms` across phases, convert to seconds. Average across sessions. The details dict includes min, max, p50, and p95 (nearest-rank percentile, not interpolated — for small datasets this can differ from the true 95th percentile).

**N/A conditions:** Returns `score=0.0` internally when no sessions have `duration_ms` data. The report detects this via `details.sessions_with_duration == 0` and displays "N/A."

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 60s | Phases complete quickly |
| Yellow | 60s–300s | Noticeable execution time |
| Red | > 300s | Phases take a long time |

---

## token_efficiency — Tokens / phase

**What it measures:** Average tokens (input + output) consumed per phase.

**What it tells you:** How efficiently each phase uses context. High token counts drive both cost and latency.

**What action it drives:** Reduce retrieved context size and tighten prompts. High token counts combined with high cost confirm that token volume is driving spend.

**How it's computed:** For each session, compute the mean of `(tokens_in + tokens_out)` across phases with token data. Then average across sessions. This is a **two-level mean** (per-phase within session, then per-session across the dataset) — not a global mean of all phase token counts. The difference: a session with 6 phases averaging 1000 tokens/phase counts the same as a session with 2 phases, even though the former consumed 3x more total tokens. This normalization makes the metric comparable across sessions with different numbers of phases.

Phases where both `tokens_in` and `tokens_out` are `None` are skipped. Phases with only one of the two set treat the missing value as 0.

**N/A conditions:** Returns `score=0.0` internally when no phases have token data. The report detects this via `details.sessions_with_tokens == 0` and displays "N/A."

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 2,000 | Lean and efficient |
| Yellow | 2,000–8,000 | Moderate token usage |
| Red | > 8,000 | Excessive tokens per phase |

---

## file_prediction_accuracy — File prediction accuracy

**What it measures:** How accurately the agent's triage-phase file list (`triage.output_structured.files`) predicts the set of files actually changed during implementation.

**What it tells you:** Whether the agent's scope assessment at triage time matches reality. A high score means the agent consistently identifies the right files at the start of a session, which improves planning reliability, review targeting, and cost forecasting. A low score suggests the scope estimate is unreliable — either missing files that will change (low recall) or predicting files that remain untouched (low precision).

**What action it drives:** Examine sessions with low F1. Repeated low recall (agent misses files it later touches) suggests the triage prompt underspecifies scope. Repeated low precision (agent predicts files it never changes) suggests over-broad scope estimates. Both indicate triage quality issues worth addressing in prompts or code-area resolution.

**How it's computed:** For each session with a non-empty `triage.output_structured["files"]` list:

1. **Predicted set** — normalise each path (strip `./`, lowercase), collect as a set.
2. **Actual set** — from `implement.output_structured["files_changed"]` (SODA format: list of dicts with `path` key) or `implement.files_modified` (Alcove format: list of strings). Falls back to `files_modified` when `files_changed` is absent or empty.
3. **Per-session F1** = `2 × precision × recall / (precision + recall)` where precision = `|predicted ∩ actual| / |predicted|` and recall = `|predicted ∩ actual| / |actual|`. Returns `0.0` when either set is empty.

**Score** = mean of per-session F1 values. Each session counts equally regardless of file count (macro-average).

**Details dict** also includes `micro_precision`, `micro_recall`, and `micro_f1` computed by pooling all true positives, predicted counts, and actual counts across sessions — useful for aggregate diagnostics.

**N/A conditions:** Returns `score=None` (displayed as N/A) when no sessions have triage file predictions. This is expected when the triage phase does not emit a `files` list or when sessions use an adapter that does not populate `output_structured`.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.70 | Agent consistently identifies the right files |
| Yellow | 0.40–0.69 | Scope estimates are partially correct |
| Red | < 0.40 | Triage file predictions are unreliable |

---

## triage_calibration -- Triage calibration

**What it measures:** Fraction of sessions where the agent's triage complexity prediction is consistent with the actual session cost.

**What it tells you:** Whether the agent's upfront complexity estimate (`small`, `medium`, or `large`) accurately predicts how expensive the session will be. A well-calibrated agent labels cheap tasks as `small` and expensive tasks as `large`, enabling accurate planning and resource allocation.

**What action it drives:** Low calibration in one direction (e.g., many `small` sessions that cost more than $8) suggests the agent is underestimating effort during triage. High calibration means you can trust the triage output for scheduling and cost forecasting.

**How it's computed:** For each session with a triage phase containing `output_structured.complexity` and a `total_cost_usd`:

- `small` → calibrated if `cost <= $8.00`
- `medium` → calibrated if `cost <= $16.00`
- `large` → always calibrated (no upper bound)

Score = `calibrated_sessions / sessions_with_triage_and_cost`. Returns N/A when no qualifying sessions exist.

**N/A conditions:** Returns `score=None` (displayed as N/A) when no sessions have both a triage complexity prediction and a cost. This is expected for sessions using adapters that do not emit a triage phase.

| Zone | Range | Meaning |
|------|-------|---------|
| Green | >= 0.80 | Triage predictions are reliable |
| Yellow | 0.60--0.79 | Moderate miscalibration |
| Red | < 0.60 | Predictions are frequently wrong |
