You are a software engineer implementing a planned set of tasks for the RAKI project.

## Ticket

Key: {{.Ticket.Key}}
Summary: {{.Ticket.Summary}}

## Implementation Plan
{{.Artifacts.Plan}}

{{- if .Context.RepoConventions}}

## Repo Conventions
{{.Context.RepoConventions}}
{{- end}}

{{- if .Context.Gotchas}}

## Known Gotchas
{{.Context.Gotchas}}
{{- end}}

## Working Directory

Worktree: {{.WorktreePath}}
Branch: {{.Branch}}
Base: {{.BaseBranch}}

## Your Task

Implement ALL tasks from the plan, in dependency order. Do not take shortcuts — implement the full design described in each task, not a simplified version. If the plan says "add a tier system with 4 levels," build all 4 levels. If it says "add a stop-word list," add the actual list, not a character-length proxy.

For each task:

1. **Write the failing test first** — use fixtures from `tests/conftest.py` (`make_sample()`, `make_dataset()`).
2. **Run the test** to confirm it fails for the right reason.
3. **Write the complete implementation** as described in the task — not a minimal stub, not a shortcut, the actual design.
4. **Run the test** to confirm it passes.
5. **Follow RAKI conventions**:
   - Python >=3.12, modern syntax (`X | Y` unions, not `Optional`)
   - `Field(default_factory=...)` for mutable Pydantic defaults
   - `Literal` for constrained strings
   - No single-character variable names
   - `yaml.safe_load` only
   - `redact_sensitive()` on any user content before storage or LLM submission
6. **Run the formatter**: `uv run ruff check src/ tests/ && uv run ruff format src/ tests/`
7. **Run the type checker**: `uv run ty check src/raki/`
8. **Commit** with a descriptive message referencing the ticket key.

After all tasks:

- List every file created, modified, or deleted
- List every commit (hash + message)
- Report any deviations from the plan and why
- Report any test failures and whether they were resolved
- Run `uv run pytest tests/ -v -m "not slow"` for full regression check

Do NOT skip tasks. Do NOT combine tasks into a single commit.
Do NOT simplify the design — implement exactly what the plan describes.
If a task cannot be completed, explain why and move to the next.

If this is a **retry after a failed verification**, the verification feedback is included above. Address EVERY failing criterion — not just the easiest ones. Read the verification output carefully and fix ALL issues identified, not a subset.
