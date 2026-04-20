## [0.6.0] — 2026-04-20

### Breaking Changes

- The ``--tenant`` CLI option has been removed. It wrote a field nothing consumed. (#81)
- ``raki report`` now takes the input file as a positional argument: ``raki report file.json`` replaces ``raki report --input file.json``.

### Features

- Implement ``--adapter`` filtering to force a specific session adapter instead of auto-detection. (#79)
- Implement ``--metrics`` filtering (comma-separated) and ``raki metrics`` subcommand for listing available metrics with ``--json`` support. (#80)
- Add phase execution time metric (``phase_execution_time``) — sums ``duration_ms`` per session with p50/p95/min/max in details. (#83)
- Add token efficiency metric (``token_efficiency``) — average tokens (in + out) per phase. (#84)
- Lower Python floor from ``>=3.14`` to ``>=3.12``. CI matrix now tests both 3.12 and 3.14. (#85)
- Add ``--judge-provider`` option for LLM provider selection (``vertex-anthropic`` or ``anthropic``). (#86)
- Add ``raki validate --deep`` for smoke-testing adapters and operational metrics without LLM calls. (#88)

### Bug Fixes

- Wire ground truth matching into CLI ``run`` and ``validate`` commands. Previously ``load_ground_truth()`` and ``match_ground_truth()`` were never called, causing Ragas metrics to silently produce 0.0 scores. (#77)
- Show "N/A (no data)" instead of misleading 0.0 for metrics when session data lacks the required fields (e.g., token counts). (#101)
- Pass ``provider="anthropic"`` to Ragas ``llm_factory()`` and remove ``top_p`` from model args to fix Anthropic API compatibility. (#104)
- Update ``--threshold`` + ``--no-llm`` warning to accurately state that threshold applies only to LLM-backed retrieval metrics.

### Documentation

- Add CI quality gate example workflow (``docs/examples/github-actions-quality-gate.yml``). (#87)

### Internal Changes

- Extract adapter registry to ``default_registry()`` in ``adapters/__init__.py``, preparing for plugin discovery in v0.7.0. (#78)
- Extract report re-rendering helpers (``MetricStub``, ``is_session_data_stripped``, ``metric_stubs_from_metadata``) to ``raki/report/rerender.py``. (#82)


## [0.5.0] — 2026-04-17

- Redesigned HTML score cards with verdict-based drill-down
- ``raki report`` command for re-rendering from saved JSON
- ``raki report --diff`` for comparing two runs

## [0.4.0] — 2026-04-15

- Security hardening: symlinks, encoding, redaction, size limits
- Adapter completeness: model_id, tokens, recursive loading, generational sorting

## [0.3.0] — 2026-04-14
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
