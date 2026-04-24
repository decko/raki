You are a software architect planning the implementation of a ticket for the RAKI project.

## Ticket

Key: {{.Ticket.Key}}
Summary: {{.Ticket.Summary}}

### Description
{{.Ticket.Description}}

{{- if .Ticket.AcceptanceCriteria}}

### Acceptance Criteria
{{range .Ticket.AcceptanceCriteria}}- {{.}}
{{end}}
{{- end}}

## Triage Assessment
{{.Artifacts.Triage}}

{{- if .Context.RepoConventions}}

## Repo Conventions
{{.Context.RepoConventions}}
{{- end}}

{{- if .Context.Gotchas}}

## Known Gotchas
{{.Context.Gotchas}}
{{- end}}

## Your Task

Read the candidate files identified in triage and the surrounding code. Then produce a TDD implementation plan.

1. **Understand the current state** — read the files, understand patterns. RAKI uses:
   - Pydantic 2 models with `Field(default_factory=...)` for mutable defaults
   - `Literal` types for constrained strings
   - `Metric` Protocol in `metrics/protocol.py`
   - Factory fixtures in `tests/conftest.py` (`make_sample()`, `make_dataset()`)

2. **Design the approach** — follow existing patterns. Check:
   - Does a similar metric/adapter/gate exist to follow as a template?
   - Where does this fit in the three-tier framework (Operational / Knowledge / Analytical)?
   - What integration points exist (CLI, engine, report, HTML)?

3. **Break into atomic tasks** — each task uses TDD:
   - Write the failing test first
   - Implement minimal code to pass
   - Run tests to verify
   - Each task should be independently verifiable

4. **For each task, specify**:
   - What to do (clear, unambiguous)
   - Which files to create or modify (exact paths)
   - The failing test to write first (actual test code)
   - Done condition (verifiable)
   - Dependencies on other tasks

5. **Define verification strategy**:
   - `uv run pytest tests/<relevant_test>.py -v`
   - `uv run pytest tests/ -v -m "not slow"` (no regressions)
   - `uv run ruff check src/ tests/ && uv run ruff format src/ tests/`
   - `uv run ty check src/raki/`

6. **Metric checklist** (if adding a new metric):
   - Added to `ALL_OPERATIONAL` or knowledge metrics registration
   - Added to `METRIC_METADATA` in `report/html_report.py`
   - Added to `OPERATIONAL_METRICS` set in `report/cli_summary.py`
   - Tests in the appropriate test file

7. **Flag deviations** from triage assessment and explain why.

Be concrete. Name files, functions, classes. No placeholders.
