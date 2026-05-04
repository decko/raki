# AGENTS.md -- Project context for AI assistants

## Project objective

**RAKI** (Retrieval Assessment for Knowledge Impact) is a standalone CLI tool that evaluates agentic RAG quality from session transcripts.

- **Success looks like**: Deterministic, reproducible evaluation of retrieval quality across session formats, with clear metrics and actionable reports.
- **Non-goals**: RAKI is not an orchestrator, not a RAG pipeline, and not a general-purpose LLM evaluation framework.

## Tech stack

- **Language**: Python >=3.12
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
  cli.py                  # Click entry points (run, validate, report, metrics, adapters)
  model/                  # Pydantic 2 domain models (EvalDataset, EvalSample, etc.)
  adapters/               # Session format adapters (any format -> EvalDataset)
    __init__.py            # default_registry() — central adapter registration
  metrics/
    ragas/                # Ragas-based retrieval metrics (pluggable, isolated)
    operational/          # Cost, rework, verify rate, severity, latency, tokens (no LLM)
  report/                 # Rich CLI + JSON + HTML (Jinja2) report generation
    rerender.py            # MetricStub, helpers for re-rendering from saved JSON
  ground_truth/           # YAML loaders + Ragas synthetic generation
tests/
  conftest.py             # Factory fixtures (not duplicated across test files)
  fixtures/               # Realistic session data for tests
  test_<module>.py        # One test file per module
changes/                  # Towncrier news fragments (one per issue)
pyproject.toml
```

## Commands reference

```bash
uv sync --python 3.12 --all-extras       # Install all dependencies
uv run pytest tests/ -v                   # Run tests (fast, no LLM)
uv run pytest tests/ -m slow -v          # Run LLM integration tests (requires credentials)
uv run ruff check src/ tests/             # Lint
uv run ruff format src/ tests/            # Format
uv run ty check src/raki/                 # Type check
uv run raki --help                        # CLI usage
uv run raki metrics                       # List all available metrics
uv run raki metrics --json                # Machine-readable metric list
uv run raki validate -m raki.yaml --deep  # Smoke-test adapters + metrics
uv run raki trends                        # Show metric trajectories over time
uv run raki trends --since 2026-01-01    # Trends from a specific date
uv run raki trends --metrics rework_cycles,cost_efficiency  # Filter to specific metrics
uv run raki trends --last 10             # Show trends for the most recent 10 runs
uv run raki trends --json                # Machine-readable trend output
uv run towncrier build --draft --version X.Y.Z  # Preview changelog
uv run towncrier build --version X.Y.Z    # Build changelog (consumes fragments)
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
- **Task agents** (Sonnet): implement code, run tests, leave changes unstaged.
- **Review agents** (Opus): review diffs, classify findings as `CRITICAL` / `IMPORTANT` / `MINOR`.
- `Assigned-by` trailer tracks delegation.
- Max 3 review iterations per task.

### Specialist dispatch matrix

| Specialist | Model | When to dispatch |
|-----------|-------|-----------------|
| Python | Opus | Every code change |
| Security | Opus | Adapters, CLI, reports, credentials |
| RAG | Opus | Ragas metrics, ground truth, LLM setup |
| UXD | Opus | CLI commands, options, error messages, output |
| CLI | Opus | Click patterns, option interactions, exit codes |
| Doc Writer | Opus | Any release, any user-facing behavior change |
| PM | Opus | Scope decisions, prioritization, feature design |

Python Specialist runs on every code issue. Always cross-reference findings between specialist rounds.

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

## Publishing

- **PyPI**: `pip install raki` — published via Trusted Publishing (OIDC), no API tokens
- **Workflow**: push a `v*` tag → `.github/workflows/publish.yml` runs tests → publishes to PyPI → creates GitHub Release with auto-generated notes
- **Manual trigger**: `workflow_dispatch` runs tests only (no publish) — useful for dry runs
- **Version lives in two places**: `pyproject.toml` and `src/raki/__init__.py` — both must match before tagging
- **README on PyPI**: `readme = "README.md"` in pyproject.toml renders the README as the project description page

### Changelog (towncrier)

Changelogs are generated from news fragments in `changes/` via towncrier.

- **Fragment naming**: `<issue-number>.<type>` (e.g., `83.feature`, `101.fix`)
- **Fragment types**: `breaking`, `feature`, `fix`, `doc`, `misc`
- **Orphan fragments**: `+<name>.<type>` for changes not tied to a single issue
- **Each PR should create a fragment** describing the user-facing change
- **At release time**: `uv run towncrier build --version X.Y.Z` generates CHANGELOG.md and removes fragments

### Release gating

**NEVER tag or push a version tag without explicit human approval.** Automated agents (orchestrators, task agents) must NOT tag releases. The release decision belongs to the project owner.

When all milestone issues are merged and CI is green, the orchestrator reports readiness and STOPS. The human owner then:

1. Reviews the diff from the last release: `git log v<previous>..main --oneline`
2. Tests against real session data
3. Decides whether to tag

### Pre-release checklist (human-driven)

1. Run `uv run pytest tests/ -v` — all tests pass
2. Run `uv run pytest tests/ -m slow -v` — LLM integration tests pass (if credentials available)
3. Run `raki run` against real session data (e.g., SODA sessions at `~/dev/soda`)
4. Open the HTML report in a browser — verify score cards, N/A display, drill-down
5. Run `raki validate --deep` — smoke-test adapters and metrics
6. Run `uv run towncrier build --version X.Y.Z` — generate changelog
7. Bump version in `pyproject.toml` and `src/raki/__init__.py`
8. Commit, tag `vX.Y.Z`, push tag — workflow handles PyPI + GitHub Release

## Documentation rule

Every spec must include doc update requirements. Every implementation task that changes user-facing behavior must update the relevant docs in the same PR. Never defer doc updates to a separate task.

## Orchestrator

Use the `/orchestrate` skill in Claude Code to coordinate milestone-level development. It dispatches tickets to the SODA pipeline, manages issue labels, verifies base health between merges, and tracks costs. SODA phase prompts live in `docs/soda/`.

## Gotchas

1. **ty is strict** -- Protocol attributes must be satisfied by class variables, not instance attributes.
2. **Ragas 0.4 + Anthropic** -- `llm_factory()` must receive `provider="anthropic"` explicitly (defaults to `"openai"`). Must pop `top_p` from `llm.model_args` after creation — Anthropic rejects `temperature` + `top_p` together. GoogleEmbeddings ignores pre-configured clients (known bug, #106).
3. **Large session files** -- some session formats can be 50MB+; `detect()` must read only the first 32KB.
4. **ReviewFinding.severity** -- `Literal["critical", "major", "minor"]`, not bare `str`.
5. **Operational metrics** -- return raw values (cost in $, cycles as count), not 0-1 normalized.
6. **No blended score** -- operational and retrieval metrics are separate categories; never combine them.
7. **`rich<15` constraint** -- instructor 1.x requires `rich>=13.7,<15`. Without this pin, uv resolves instructor 0.4.0 which pulls docstring-parser 0.15 (broken on Python 3.14 due to removed `ast.NameConstant`).
8. **Version in two places** -- `pyproject.toml` and `src/raki/__init__.py` must match. Bump both before tagging.
9. **HTML report is optional** -- jinja2 is in the `[html]` extra. Template uses `autoescape=True` (XSS protection). `METRIC_METADATA` dict must stay in sync with metric classes — `TestMetricMetadataSync` enforces this.
10. **Test before tagging** -- always run pytest + manual verification against real data + open the HTML report in a browser before pushing a version tag.
11. **New metric checklist** -- every new metric must be added to 3 places: `ALL_OPERATIONAL` in `metrics/operational/__init__.py`, `METRIC_METADATA` in `report/html_report.py`, and `OPERATIONAL_METRICS` in `report/cli_summary.py`. Missing any one causes silent misclassification (e.g., threshold gating treats it as retrieval).
12. **N/A display convention** -- metrics signal "no data" via `details` dict: `sessions_with_*: 0` for missing session fields, `skipped: "<reason>"` for Ragas metrics without ground truth. Renderers check these to show "N/A (reason)" instead of misleading 0.0.
13. **Ground truth matching is fragile** -- `match_ground_truth()` uses `code_area` domain-token overlap from triage phases only. Sessions without a triage phase will not match. Low match rate triggers a CLI warning.
14. **`scikit-network` requires C++ compiler** -- the `ragas` extra pulls `scikit-network` which needs `g++`. Documented in getting-started.md.
15. **`max_tokens=4096` on `llm_factory()`** -- Ragas defaults to `max_tokens=1024` for LLM output. Structured output via instructor overflows at 1024. All `llm_factory()` calls must pass `max_tokens=4096`.
16. **Three-tier metric categories** -- Operational (zero config, in `ALL_OPERATIONAL`), Knowledge (`--docs-path`, in `ALL_KNOWLEDGE`), Analytical (`--judge`, Ragas metrics). Each category has its own CLI section header and registration path.
17. **Knowledge matching is hybrid** -- `_common.py` uses path matching (finding file vs chunk domain) + word matching (>=3 non-stop-word overlap) with confidence tiers (strong/domain/content/None). Only strong + domain tiers count as "covered." Do not revert to simple word overlap.
18. **`--gate` is the canonical threshold flag** -- not `--threshold`. The old float-only `--threshold` is deprecated. All docs and examples use `--gate 'metric>value'`.
19. **Google provider has silent 0.0 bug** -- instructor #1658 causes Gemini structured output to silently return default values. `is_instructor_silent_zero()` in `adapter.py` detects this. Affected sessions are excluded from the metric average with a warning.
20. **Soda implement prompt must say "complete"** -- TDD "minimal implementation" instruction causes agents to shortcut complex designs (tweak a constant instead of building the full system). The soda implement prompt says "Write the complete implementation as described in the task."
21. **SODA rework short-circuits** -- when verify fails, the rework implement agent often sees existing code and declares "already implemented" without fixing the verify feedback (soda#395). After verify failure, check if rework implement spent <2% of first-pass budget — if so, go straight to manual fix in the worktree.
22. **Report config key names** -- judge config fields use `llm_` prefix: `llm_provider`, `llm_model`, `llm_temperature`, `llm_max_tokens`. Not `temperature`, not `batch_size`. All are None when `skip_judge=True`.
23. **History log at `.raki/history.jsonl`** -- created automatically by `raki run`. Must be in `.gitignore`. `HistoryEntry.metrics` is `dict[str, float]` — absent keys mean metric not computed, not None or 0.0.
24. **Ticket budget limit** -- SODA tickets should be scoped under 100K tokens. Use the `/scope` skill to estimate. Split tickets that exceed the budget (core logic vs CLI wiring is a natural boundary).
25. **Towncrier fragment types** -- only `.breaking`, `.feature`, and `.fix` are configured. Using `.change` or other suffixes gets silently ignored by `towncrier build`.
26. **Manifest symlink path escape** -- manifest session paths cannot be symlinks to external directories. The real path is resolved and checked against project root. Copy sessions into the manifest directory instead of symlinking.
27. **`soda-system-prompt-*.md` stray files** -- SODA sometimes leaves temporary prompt files in the repo root. These are in `.gitignore` and should never be committed.
28. **`--docs-path` must match the project being worked on** -- when evaluating SODA sessions that implement raki code, point `--docs-path` to raki's docs, not SODA's. When evaluating Alcove sessions working on pulp-service, point to pulp-service docs. Wrong docs → knowledge_gap_rate=1.00 because findings reference source files the docs don't cover.
29. **`vertex-anthropic` is the reliable judge provider** -- `anthropic` needs `ANTHROPIC_API_KEY` (not set by default). `google` has an async/sync client mismatch (#233). `answer_relevancy` requires either `GOOGLE_CLOUD_PROJECT` or `VERTEXAI_PROJECT`; `create_ragas_embeddings` normalises `GOOGLE_CLOUD_PROJECT` from `VERTEXAI_PROJECT` so `GoogleEmbeddings` can find the project internally (#231). Default to `--judge-provider vertex-anthropic --judge-model claude-sonnet-4-6`.

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
- Bumping version in only one of the two files (pyproject.toml / __init__.py).
- Tagging a release without testing against real data first.
- Tagging or pushing version tags autonomously — releases require explicit human approval.
- Deferring doc updates to a separate task instead of shipping with the code.
- Adding a metric to `ALL_OPERATIONAL` but forgetting `METRIC_METADATA` or `OPERATIONAL_METRICS`.
- Showing 0.0 for metrics with no data instead of N/A (check `sessions_with_*` / `skipped` keys).
- Calling `llm_factory()` without `provider="anthropic"` (defaults to openai, fails).
- Not creating a towncrier fragment for user-facing changes.
- Implementing only the easiest parts of a multi-task plan and skipping the complex ones (build the full design, not a shortcut).
- Using old metric names (`first_pass_verify_rate` was renamed to `first_pass_success_rate` in v0.8.0).
- Passing all doc chunks to Ragas instead of selecting top-N relevant ones per session.
- Not testing against real soda data (`~/dev/soda/.soda/`) before claiming a feature works. Unit tests verify code, not features.
- Using `temperature` or `batch_size` as report config keys instead of `llm_temperature` / `llm_max_tokens`.
- Treating absent metric keys in `HistoryEntry.metrics` as 0.0 instead of "no data" (gaps, not zeros).
