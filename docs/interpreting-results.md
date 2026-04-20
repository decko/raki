# Interpreting Results

How to read the RAKI HTML report and understand what the metrics mean.

For the full metric reference with zone tables and red-zone actions, see
[interpretation-reference.md](interpretation-reference.md).

## Operational Health

### Verify Rate

**What it measures:** The fraction of sessions where the agent's work passed
all verification checks on the first attempt.

- **Target:** >85%
- **Direction:** Higher is better
- A high verify rate means the agent consistently delivers correct work
  without needing rework.

### Rework Cycles

**What it measures:** The average number of review-fix iterations per session.

- **Good:** <1.0
- **Direction:** Lower is better
- **Color thresholds:** Green (<1.0), Yellow (1.0--2.0), Red (>2.0)

### Cost per Session

**What it measures:** The average LLM API cost per session in USD.

- The score card shows the average cost and, when per-session data is
  available, the min--max range across all sessions.

### Review Findings

**What it measures:** The number and severity of issues found by reviewers
across all evaluated sessions.

#### Severity Distribution

Instead of a single numeric score, the report shows a **stacked distribution
bar** with counts for each severity level (critical, major, minor) and a
traffic-light label:

| Label | Condition |
|-------|-----------|
| **Clean** | 0 critical + 0 major findings |
| **Minor** | 0 critical, but some major findings |
| **Moderate** | Some critical findings, but weighted severity <= 0.5 |
| **Severe** | Weighted severity > 0.5 |

The **weighted severity** formula is:

    weighted = (3 * critical + 2 * major + 1 * minor) / (3 * total)

This weights critical findings three times more than minor ones. A "Severe"
label means a disproportionate number of findings are critical or major.

### Knowledge Miss Rate

**What it measures:** How often rework happened because the agent lacked the
right reference material in its retrieved context.

- **Target:** <0.20
- **Direction:** Lower is better
- This metric is **hidden** when no sessions have `knowledge_context` data.
  A footnote explains: "Knowledge Miss Rate omitted -- no retrieval context
  available in sessions."

## Retrieval Quality

These metrics require LLM-backed evaluation (they are hidden in `--no-llm`
mode, with a footnote explaining why).

### Context Precision

**What it measures:** How much of what the retriever pulled in was actually
relevant to the question.

- **Target:** >0.80
- **Direction:** Higher is better

### Context Recall

**What it measures:** How much of the needed information the retriever
successfully found.

- **Target:** >0.80
- **Direction:** Higher is better

### Faithfulness

**What it measures:** How closely the agent's output sticks to the facts in
its source material.

- **Direction:** Higher is better
- This metric is **experimental** for agentic sessions.

## Comparing Runs

Use `raki report --diff baseline.json compare.json` to compare two evaluation
runs side by side. The diff output highlights what changed between runs.

### Reading the Diff Output

**Coverage line** shows how many sessions were matched by `session_id`:

```
Matched: 42/58 sessions (16 new, 4 dropped)
```

If sessions were dropped or added, a warning banner notes that aggregate
deltas are based on matched sessions only.

**Aggregate deltas** show the change for each metric:

- **Direction indicators**: a green up arrow (improved) or red down arrow
  (regressed) respects each metric's `higher_is_better` setting. For
  example, a *decrease* in rework cycles is shown as an improvement.
- **Flat deltas** are shown with an equals sign when values are identical.

**Session transitions** group sessions by verdict change:

- Regressions (PASS to REWORK, PASS to FAIL) are listed first.
- Improvements (REWORK to PASS, FAIL to PASS) follow.
- Sessions with unchanged verdicts are not listed.

### What You Need

Both reports must have been generated with `--include-sessions` for
per-session transition analysis. Without session data, only aggregate
metric deltas are shown, and a warning notes that per-session comparison
is unavailable.

### When to Use Diff

- **After prompt or config changes**: compare before/after runs to verify
  that changes improved metrics without introducing regressions.
- **Across evaluation batches**: compare runs on different session sets to
  spot trends in retrieval quality or operational health.
- **CI pipelines**: use diff to gate merges -- if regressions appear, the
  diff output shows exactly which sessions regressed and by how much.
