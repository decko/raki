## [0.9.0] — 2026-04-24

### Features

- Append a compact JSONL entry to ``.raki/history.jsonl`` after every ``raki run`` so metric trends can be tracked across evaluation runs. Each line records ``run_id``, ``timestamp``, ``sessions_count``, ``metrics``, ``config_hash``, ``git_sha``, and ``manifest``. Use ``--history-path PATH`` to write to a custom location, or ``--no-history`` to skip the log entirely. (#170)
- Add ``raki trends`` command — show metric trajectories over time from the JSONL history log.

  ``raki trends`` reads ``.raki/history.jsonl`` and displays a sparkline + delta table for every metric, grouped by tier (Operational → Knowledge → Analytical). Key options:

  - ``--metrics NAMES`` — comma-separated filter to specific metrics (validates against known names)
  - ``--since DATE`` / ``--until DATE`` — restrict to a time window (YYYY-MM-DD)
  - ``--last N`` — cap to the most recent N evaluation runs
  - ``--json`` — machine-readable output with full value series and delta
  - ``--history-path PATH`` — point to a custom history file

  Metric names from older raki versions are automatically translated (e.g. ``first_pass_verify_rate`` → ``first_pass_success_rate``). Runs that did not record a given metric are silently skipped (gap handling).

  (#171)
- Serialize judge configuration fields (``llm_provider``, ``llm_model``, ``llm_temperature``, ``llm_max_tokens``) into the report JSON ``config`` dict. When ``skip_llm`` is true, all judge fields are ``None``. Old reports without these fields load without error. (#173)
- Warn when judge configurations differ in ``--diff`` comparison. When ``raki report --diff`` compares two reports that used different LLM judge settings (model or provider), a yellow warning is now shown so users know the retrieval quality scores may not be directly comparable. (#187)

### Bug Fixes

- Fix Google judge provider silently returning 0.0 scores when ``instructor`` (issue #1658) fails to parse structured output. Affected sessions are now skipped with a warning and the metric returns ``score=None`` instead of a misleading zero average. Also removes ``top_p`` from Google LLM ``model_args`` to prevent API rejection when both ``temperature`` and ``top_p`` are set. (#169)
- Alcove adapter now detects ``rework_cycles`` and ``total_phases`` from transcript tool calls instead of hardcoding 0 and 1. A rework cycle is counted when a test/lint command fails, the agent edits a previously-written file, then re-runs the test. Multi-phase detection groups tool calls into analysis (Read/Grep), coding (Write/Edit), and testing (Bash test runners) phases. Explicit ``rework_cycles`` and ``phases`` values in bridge-format JSON take priority over transcript detection. TDD workflows (test-first, then implement new files) correctly produce zero rework cycles. (#176)


## [0.8.0] — 2026-04-23

### Breaking Changes

- The `first_pass_verify_rate` metric has been renamed to `first_pass_success_rate`
  and its algorithm has changed.

  **Old behaviour:** Counted sessions where the `verify` phase had `generation=1`
  and `status=completed`. This could show 1.00 even when `rework_cycles` showed
  a non-zero average, producing contradictory signals.

  **New behaviour:** Counts sessions where `session.rework_cycles == 0`. The metric
  is now guaranteed to be consistent with `rework_cycles`.

  **Migration:**

  - Replace `first_pass_verify_rate` with `first_pass_success_rate` in any
    `--gate` expressions, manifests, or saved report files you reference.
  - Python code importing `FirstPassVerifyRate` should import `FirstPassSuccessRate`
    from `raki.metrics.operational.verify_rate` instead.
  - The class `FirstPassVerifyRate` has been removed; `FirstPassSuccessRate`
    is its replacement.
  - Empty datasets now return `score=None` (was `0.0`) — consistent with
    `SelfCorrectionRate`.

  (#150)

### Features

- AlcoveAdapter now supports the bridge/alcove session format (id + task_id + transcript) in addition to the classic format (session_id + transcript). Bridge fields like task_name, status, and started_at are mapped to SessionMeta. (#163)

### Bug Fixes

- Fixed ``--docs-path`` guard using ``manifest_file.parent`` as the project root
  instead of the current working directory.  Docs paths within CWD but outside the
  manifest's parent directory (e.g. repo-root ``docs/`` with the manifest in a
  ``config/`` subdirectory) are now accepted correctly. (#131)
- Fix CLI output to show three-tier section headers (Operational Health / Knowledge Quality / Retrieval Quality) and progression nudges guiding users toward the next metric tier. (#133)
- ``--gate`` now rejects completely unknown metric names with exit code 2 and a helpful error listing valid metrics, instead of silently skipping the gate check. (#136)
- Knowledge metrics (knowledge_gap_rate, knowledge_miss_rate) now appear in raki metrics and raki metrics --json output. (#138)
- ``raki report`` now accepts ``--gate`` and ``--require-metric`` flags (matching ``raki run``), allowing per-metric quality gates to be applied against a saved JSON report without re-running the evaluation. Also adds ``-q``/``--quiet`` to suppress summary and gate output. (#139)
- Quality gate output now rounds actual values to 4 decimal places instead of showing raw floats. (#141)
- Knowledge matching now uses hybrid path + word matching with stop-word filtering and confidence tiers (strong/domain/content). Requires >=3 non-stop-word overlap instead of single word match. Produces actionable gap and miss rates on real data. (#151)

### Documentation

- Added detailed `rationale` attribute to every non-Ragas metric class (7 operational + 2 knowledge) and exposed it in `raki metrics --json` output. Added new `docs/metrics/rationale-and-interpretation.md` with per-metric design rationale, interpretation tables, pitfall warnings, and combined metric pattern analysis. Cross-referenced from `docs/metrics/operational.md`, `docs/metrics/knowledge.md`, and `docs/interpretation-reference.md`. (#152)


## [0.7.1] — 2026-04-22

### Features

- Add ``google`` as a third LLM judge provider, enabling Gemini models via Vertex AI (``--judge-provider google``). (#110)
- Invert CLI default: ``raki run`` now skips LLM metrics by default. Use ``--judge`` to enable LLM-judged analytical metrics. ``--no-llm`` is deprecated and will be removed in v0.8.0. (#112)
- Decompose ``knowledge_miss_rate`` into three metrics: ``self_correction_rate`` (operational), ``knowledge_gap_rate`` and ``knowledge_miss_rate`` (knowledge tier). All return N/A when their denominator is zero. (#113)
- Faithfulness and answer relevancy now work without ground truth by synthesizing retrieval context from session transcripts. Results from synthesized context are tagged with ``(inferred)`` in CLI output. (#114)
- Add ``--docs-path`` to load project documentation for knowledge metrics (``knowledge_gap_rate``, ``knowledge_miss_rate``). Supports format-aware chunking (Markdown, RST, plaintext) with symlink rejection and size limits. (#115)
- Add per-metric quality gates (``--gate 'metric>value'``) and CI regression detection (``--fail-on-regression``). Distinct exit codes: 0=clear, 1=threshold, 3=regression, 4=both. (#116)

### Bug Fixes

- Wire doc chunks from --docs-path as reference_contexts for context precision/recall. (#129)
- Use per-domain matching for knowledge gap and miss rates. (#130)
- Pin instructor>=1.0 to fix LLM judge crashes at runtime. (#134)
- Truncate synthesized contexts and response text to prevent Ragas max_tokens errors. Extract triage/plan summary as response instead of raw code output. Select top-10 relevant doc chunks per session by keyword overlap. (#135)
- Handle --require-metric gracefully when metric is not computed. (#137)
- Store N/A metrics as null in JSON reports instead of misleading 0.0. (#140)

### Documentation

- Overhaul documentation with three-tier metric framework (Operational, Knowledge, Analytical) and CI integration guide. (#117)


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
