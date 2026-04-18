# AGENTS.md -- Project context for AI assistants

## Project objective

**RAKI** (Retrieval Assessment for Knowledge Impact) is a standalone CLI tool that evaluates agentic RAG quality from session transcripts.

- **Success looks like**: Deterministic, reproducible evaluation of retrieval quality across session formats, with clear metrics and actionable reports.
- **Non-goals**: RAKI is not an orchestrator, not a RAG pipeline, and not a general-purpose LLM evaluation framework.

## Tech stack

- **Language**: Python 3.14
- **Package manager**: uv (not pip, not poetry, not conda)
- **Build backend**: hatchling
- **Models**: Pydantic 2 (`Field(default_factory=...)` for mutable defaults, `Literal` for constrained strings)
- **CLI**: Click
- **Terminal output**: Rich
- **Linting/formatting**: ruff (line-length 100)
- **Type checking**: ty (not mypy)
- **Testing**: pytest, pytest-asyncio for async tests
- **Metrics**: Ragas (pluggable, isolated in `metrics/ragas/`) + operational metrics (no LLM required)
- **Reports**: Rich CLI summary + JSON + HTML (Jinja2)
- **Ground truth**: curated YAML + Ragas synthetic generation

## Repository layout

```
src/raki/
  __init__.py
  cli.py                  # Click entry points
  model/                  # Pydantic 2 domain models (EvalDataset, EvalSample, etc.)
  adapters/               # Session format adapters (any format -> EvalDataset)
  metrics/
    ragas/                # Ragas-based retrieval metrics (pluggable, isolated)
    operational/          # Cost, rework cycles, verify rate, severity (no LLM)
  report/                 # Rich CLI + JSON + HTML (Jinja2) report generation
  ground_truth/           # YAML loaders + Ragas synthetic generation
tests/
  conftest.py             # Factory fixtures (not duplicated across test files)
  fixtures/               # Realistic session data for tests
  test_<module>.py        # One test file per module
pyproject.toml
```

## Commands reference

```bash
uv sync --python 3.14 --all-extras       # Install all dependencies
uv run pytest tests/ -v                   # Run tests
uv run ruff check src/ tests/             # Lint
uv run ruff format src/ tests/            # Format
uv run ty check src/raki/                 # Type check
uv run raki --help                        # CLI usage
```

## Architecture

```
Session Transcript (any format)
  -> Adapter (detects format, calls redact_sensitive())
    -> EvalDataset (internal Pydantic model)
      -> Metrics Engine
        -> Ragas metrics (retrieval quality, requires LLM)
        -> Operational metrics (cost, rework, verify rate -- no LLM)
      -> Report Generator
        -> CLI summary (Rich) | JSON | HTML (Jinja2)
```

Key design rules:
- **Pluggable adapters**: each session format gets its own adapter; all must call `redact_sensitive()` before populating `EvalSample`.
- **Metrics isolation**: Ragas lives in `metrics/ragas/` and is never imported elsewhere. Operational metrics are separate and independent.
- **No blended scores**: operational and retrieval metrics are separate categories. There is no "overall score" combining them.

## Conventions

### Development methodology

- **TDD**: Write failing tests first, then implement. No exceptions.
- **Small steps**: Prefer small, reviewable changes over large sweeps.

### Code style

- ruff enforces formatting and linting (line-length 100).
- ty for type checking (strict mode -- Protocol attributes satisfied by class variables, not instance attributes).
- No single-character variable names. Use descriptive names in loops and comprehensions.
- `yaml.safe_load` only, never `yaml.load`.

### Models

- Pydantic 2 for all domain models.
- `Field(default_factory=...)` for mutable defaults (lists, dicts, sets).
- `Literal` for constrained string fields (e.g., severity levels).

## Git workflow

- **NEVER** commit to main directly. All work happens on feature branches.
- **Branch naming**: `task/<N>-<short-name>`
- **Worktree-based development**: all implementation happens in `.worktrees/`
- **Commit format**: `<type>(<scope>): <subject>` with body explaining *why*.

### Worktree management

- All feature work in worktrees under `.worktrees/`.
- Create: `git worktree add .worktrees/task-<N> -b task/<N>-<short-name>`
- After merge: `git worktree remove .worktrees/task-<N> && git branch -d task/<N>-<short-name>`
- Before starting: check `git worktree list` and clean up stale entries.
- Never have more than 2 active worktrees at once.

### Required trailers

```
Assisted-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```
For orchestrator-assigned work, also include:
```
Assigned-by: <developer-or-orchestrator>
```

### Safety rules

- No force-push, no `--no-verify`, no amending pushed commits.
- Pre-commit hooks enforce: ruff check, ruff format, ty check, email validation (`decko@redhat.com`).

## Subagent conventions

- **Orchestrator** spawns task agents and review agents.
- **Task agents**: implement code, run tests, leave changes unstaged.
- **Review agents**: review diffs, classify findings as `CRITICAL` / `IMPORTANT` / `MINOR`.
- `Assigned-by` trailer tracks delegation.
- Max 3 review iterations per task.

## Testing conventions

- Test files: `tests/test_<module>.py`.
- Factory fixtures live in `conftest.py` (never duplicated across test files).
- Fixtures use realistic data shapes from real session formats.
- Slow tests (requiring LLM calls) marked with `@pytest.mark.slow`.
- Tests must be deterministic -- no time-dependent assertions.
- No single-character variable names in test code either.

## Security

- **Redaction**: all adapters must call `redact_sensitive()` before populating `EvalSample`.
- **YAML**: `yaml.safe_load` only, never `yaml.load`.
- **Path traversal**: manifest paths validated as descendants of project root.
- **Reports**: strip raw session data by default (`--include-sessions` to opt in).
- **No credentials** in code, config, or reports.

## Gotchas

1. **ty is strict** -- Protocol attributes must be satisfied by class variables, not instance attributes.
2. **Ragas 0.4 API** -- imports from `ragas.metrics.collections`; `ascore()` takes `SingleTurnSample`, not kwargs.
3. **Large session files** -- some session formats can be 50MB+; `detect()` must read only the first 4KB.
4. **ReviewFinding.severity** -- `Literal["critical", "major", "minor"]`, not bare `str`.
5. **Operational metrics** -- return raw values (cost in $, cycles as count), not 0-1 normalized.
6. **No blended score** -- operational and retrieval metrics are separate categories; never combine them.

## Things agents often get wrong here

- Using pip/poetry/conda instead of uv.
- Using mypy instead of ty.
- Committing directly to main instead of using worktrees.
- Writing implementation before failing tests (violates TDD).
- Calling `yaml.load` instead of `yaml.safe_load`.
- Forgetting `redact_sensitive()` in adapter code.
- Using single-character variable names in loops.
- Blending operational and retrieval metrics into one score.
- Using `str` instead of `Literal` for constrained fields.
