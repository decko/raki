# Contributing to RAKI

Thanks for your interest in RAKI! This guide covers the workflow for contributing changes.

## Getting Started

1. Fork the repository and clone your fork.
2. Set up your development environment:

```bash
uv sync --python 3.14 --all-extras
```

## Branch Naming

Create a branch from `main` using the pattern:

```
task/<issue-number>-<short-name>
```

For example: `task/42-fix-verify-rate`

## Making Changes

RAKI follows test-driven development. Write a failing test first, then implement.

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Style

Lint and format with ruff (line-length 100):

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Type Checking

```bash
uv run ty check src/raki/
```

## Pull Request Process

1. Ensure all tests pass and linting is clean before opening a PR.
2. Write a clear PR description explaining **what** changed and **why**.
3. Reference the related issue (e.g., "Closes #42").
4. Keep changes small and focused — one concern per PR.

## Conventions

- **Models**: Pydantic 2 with `Field(default_factory=...)` for mutable defaults.
- **Security**: all adapters must call `redact_sensitive()` before populating `EvalSample`.
- **YAML**: use `yaml.safe_load` only, never `yaml.load`.
- **Variable names**: no single-character variables, even in loops.
- **Metrics**: operational and retrieval metrics are never blended into a single score.

## Commit Format

```
<type>(<scope>): <subject>
```

Include a body explaining *why* the change was made, not just what changed.

## Questions?

Open an issue or start a discussion — we are happy to help.
