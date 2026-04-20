# Getting Started with RAKI

Evaluate your agentic RAG sessions in under 10 minutes.

## Prerequisites

- **Python 3.14** or later
- **[uv](https://docs.astral.sh/uv/)** package manager

## Install

```bash
git clone https://github.com/decko/raki.git
cd raki
uv sync --python 3.14 --extra html
```

> **LLM-backed metrics**: if you plan to use Ragas retrieval metrics, install
> with `uv sync --python 3.14 --all-extras` instead. This pulls in
> `scikit-network`, which requires a C++ compiler.

Verify the install:

```bash
uv run raki --help
```

## Quick Start

RAKI ships with example session data so you can try it immediately.

Validate the example manifest:

```bash
uv run raki validate --manifest examples/raki-minimal.yaml
```

Run an evaluation using only operational metrics (no LLM required):

```bash
uv run raki run --manifest examples/raki-minimal.yaml --no-llm
```

### Discovering Available Metrics

List all metrics RAKI can compute, including which ones require an LLM provider:

```bash
uv run raki metrics
```

Use `--json` for machine-readable output:

```bash
uv run raki metrics --json
```

Run only specific metrics by name (use the internal name from `raki metrics`):

```bash
uv run raki run --manifest examples/raki-minimal.yaml --no-llm --metrics cost_efficiency,rework_cycles
```

## Prepare Your Data

RAKI evaluates session transcripts produced by agentic coding tools. Each session is a directory containing phase files (`triage.json`, `implement.json`, `verify.json`, `review.json`) and an `events.jsonl` log. See `examples/sessions/` for working examples.

If your tool produces a different format, you can write a custom adapter. See the [Adapter Authoring Guide](adapter-guide.md) for details.

## Understanding the Output

The Quick Start command above produces output like this:

```
Loading sessions from examples/sessions...
Loaded 6 sessions (0 skipped, 0 errors)

Operational Health
  Verify rate                         0.75  (% sessions passing verify on first try)
  Rework cycles                       0.2  (Mean review-rework iterations per session)
  Severity score                      0.39  (Weighted severity of review findings (1.0 = no findings))
  Cost / session                      $10.93  (Mean USD cost per session)
  Knowledge miss rate                 1.00  (Ratio of rework findings caused by missing knowledge retrieval)
  ⚠ Small sample size (n=6) — scores are directional, not definitive
```

Key metrics to watch:

- **Verify rate** (0.75) -- 75% of sessions passed verification on the first attempt. Higher is better; values below 0.5 suggest systemic quality issues in the implement phase.
- **Rework cycles** (0.2) -- sessions needed an average of 0.2 review-rework iterations. Lower is better; values above 1.0 mean most sessions require at least one rework round.
- **Severity score** (0.39) -- weighted severity of review findings, where 1.0 means no findings at all. Higher is better; low values indicate reviewers are catching serious issues.

## Re-rendering Reports

RAKI saves evaluation results as JSON. You can re-render the CLI summary or generate an HTML report from a saved JSON file at any time:

```bash
# Re-render CLI summary from a saved report
uv run raki report --input results/raki-report-20260410T120000.json

# Generate an HTML report from the same JSON
uv run raki report --input results/raki-report-20260410T120000.json --html results/report.html
```

If the JSON report was generated without `--include-sessions`, RAKI will show aggregate scores only and warn that per-session drill-down is unavailable.

## Comparing Runs

Compare two evaluation runs side by side to see what improved and what regressed:

```bash
# Compare a baseline and a new run
uv run raki report --diff results/baseline.json results/compare.json

# Generate an HTML diff report
uv run raki report --diff results/baseline.json results/compare.json --html results/diff.html

# Write HTML to a specific output directory
uv run raki report --diff results/baseline.json results/compare.json -o results/
```

The diff output shows:

- **Coverage line** -- how many sessions matched between runs, plus counts of new and dropped sessions.
- **Aggregate deltas** -- metric-by-metric comparison with direction indicators (green for improvements, red for regressions).
- **Session transitions** -- which sessions changed verdict (e.g., REWORK to PASS), grouped by transition type with regressions listed first.

For per-session transition analysis, both reports must have been generated with `--include-sessions`. Without session data, only aggregate deltas are shown.

## Next Steps

- [Results Interpretation Reference](interpretation-reference.md) -- thresholds, patterns, and what to do about low scores
- [Ground Truth Curation Guide](curation-guide.md) -- writing effective ground truth for Ragas retrieval metrics
- [Adapter Authoring Guide](adapter-guide.md) -- integrating session formats beyond the built-in adapters
