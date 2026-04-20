# Changelog

## [Unreleased] (v0.6.0) — Make It Real

### Breaking Changes

- **`raki report` positional argument**: `raki report --input file.json` is now
  `raki report file.json`. The `--input` / `-i` option has been replaced with a
  positional argument. Update any scripts or CI pipelines that use the old syntax.
- **`--tenant` removed**: The `--tenant` CLI option has been removed entirely.
  It wrote a field nothing consumed. It can be re-added with actual functionality
  if needed later.

### Changed

- `--threshold` + `--no-llm` warning updated from generic message to:
  "No retrieval metrics active — threshold applies only to LLM-backed metrics.
  Operational metrics use non-0-1 scales; per-metric thresholds planned for v0.7.0."

## [Unreleased] (v0.4.0) — Security & Data Completeness
- Security hardening: symlinks, encoding, redaction, size limits
- Adapter completeness: model_id, tokens, recursive loading, generational sorting

## [Unreleased] (v0.3.0) — Report & CLI Polish
- Comprehensive HTML report UX improvements (colors, sessions count, display names)
- CLI fixes: descriptions, display names, threshold test, quiet validate
- Sample-level results in reports
- NoneType fix for SODA metrics

## v0.2.1
- Ragas temperature forwarding and Vertex AI embedding fixes
- Except syntax fix for Python 3.14
- Ragas SingleTurnSample compatibility fix
- JSON stdout output fix

## v0.1.0
- Core Pydantic models (EvalSample, SessionMeta, PhaseResult, etc.)
- Adapter protocol with session-schema and Alcove adapters
- Operational metrics (verify rate, rework cycles, severity, cost, knowledge miss rate)
- Ragas integration (context precision/recall, faithfulness, answer relevancy)
- Ground truth loader and domain-based matcher
- Manifest loader with path traversal protection
- CLI with run, validate, and adapters commands
- HTML report generation
