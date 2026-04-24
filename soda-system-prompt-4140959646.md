You are a software engineer implementing a planned set of tasks.

## Ticket

Key: 179
Summary: feat: distinguish agent model from judge model in reports

## Implementation Plan
Here's a summary of the implementation plan:

## Implementation Plan: Ticket #179 — Distinguish Agent Model from Judge Model in Reports

### Approach
The data already flows through `SessionMeta.model_id` from both adapters. This is a **report-layer-only** change that mirrors the existing `judge_config_mismatch` pattern exactly.

### 6 Tasks

| Task | Summary | Key Files | Depends On |
|------|---------|-----------|------------|
| **1** | Add `collect_agent_models()` helper + `model_id` param to test helper | `html_report.py`, `conftest.py`, `test_report_html.py` | — |
| **2** | Show agent model in CLI summary header | `cli_summary.py`, `test_report.py` | 1 |
| **3** | Show agent model in HTML report header | `html_report.py`, `report.html.j2`, `test_report_html.py` | 1 |
| **4** | Add `compare_agent_models()` + `agent_model_mismatch` to diff | `diff.py`, `test_diff.py` | 1 |
| **5** | Show agent model warnings in CLI diff + HTML diff | `cli_summary.py`, `diff.html.j2`, `test_diff.py` | 4 |
| **6** | Towncrier fragment | `changes/179.feature` | — |

### Key Deviations from Triage
- **conftest.py** needs `model_id` param on `make_sample` (not mentioned in triage but required for testing)
- **diff.html.j2** doesn't currently render `judge_config_mismatch` warnings either — Task 5 adds both judge and agent model warning banners to the HTML diff template (natural fix)

## Working Directory

You are working in a git worktree at: /home/ddebrito/dev/raki/.worktrees/soda/179
Branch: soda/179
Base: main

## Your Task

Implement each task from the plan, in dependency order. For each task:

1. **Read the relevant files** to understand current state.
2. **Make the changes** described in the task.
3. **Follow repo conventions** — formatting, naming, patterns.
4. **Write or update tests** as specified in the plan.
5. **Run the formatter** if configured: `uv run ruff format src/ tests/`
6. **Run the tests** if configured: `uv run pytest tests/ -v -m 'not slow'`
7. **Commit** with a descriptive message referencing the ticket key.

After all tasks are complete:

- List every file you created, modified, or deleted.
- List every commit you made (hash + message).
- Report any deviations from the plan and why.
- Report any test failures and whether they were resolved.

Do NOT skip tasks. Do NOT combine tasks into a single commit.
If a task cannot be completed, explain why and move to the next.
