# SODA Pipeline Quality Gate

Use RAKI's `gate-check` command to evaluate SODA pipeline session quality as a feedback loop.
Run periodic (per-milestone) evaluations on accumulated SODA sessions to surface quality trends
and detect regressions before they compound.

## Overview

RAKI evaluates SODA pipeline sessions and produces actionable metrics:

- **`first_pass_success_rate`** — fraction of sessions that verify on the first attempt. Low values signal prompt quality or task complexity issues.
- **`rework_cycles`** — average rework iterations per session. High values indicate implementation accuracy problems.
- **`review_severity_distribution`** — distribution of review finding severities. Tracks whether code quality is improving.
- **`cost_efficiency`** — average cost per session in USD. Useful for budget planning and catching runaway sessions.

The `gate-check` command applies threshold gates to an already-generated JSON report, making it fast to check quality without re-running the full evaluation.

## Prerequisites

- RAKI installed: `uv pip install raki`
- SODA session transcripts available (under `.soda/` or a configured path)
- A prior `raki run` to generate the JSON report to evaluate

The history log (`#170`) and `raki trends` (`#171`) are recommended companions for cross-milestone tracking. See [CI Integration Guide](ci-integration.md) for general gating patterns.

## Workflow

### Step 1: Evaluate accumulated SODA sessions

After a milestone's SODA sessions complete, run `raki run` to generate the evaluation report:

```bash
uv run raki run -m examples/soda-gate.yaml -o results/
```

This loads sessions from `.soda/` (or the path in your manifest), computes operational metrics, and writes `results/raki-report-<timestamp>.json`.

### Step 2: Check gates with SODA defaults

Run `gate-check` on the generated report to check against the built-in SODA thresholds:

```bash
uv run raki gate-check results/raki-report-*.json
```

Default thresholds (applied when no `--gate` or manifest `thresholds:` is given):
- `first_pass_success_rate>=0.6`
- `rework_cycles<=0.5`

Exit code 0 = all pass. Exit code 1 = at least one gate failed.

### Step 3: Compare against a previous milestone (optional)

Check for regressions versus the previous milestone's report:

```bash
uv run raki gate-check results/current.json --baseline results/previous.json
```

If any metric regressed beyond the noise margin (2%), `gate-check` exits with code 3.
Exit code 4 means both a threshold violation and a regression were detected.

### Step 4: Detailed metric comparison

For a more detailed view of what changed between milestones, use `raki report --diff`:

```bash
uv run raki report --diff results/previous.json results/current.json
```

This shows per-metric deltas, verdict transitions, and a direction indicator (▲/▼) for each metric.

### Step 5: Visualize metric trajectories

Use `raki trends` to see how metrics have moved over multiple milestones:

```bash
uv run raki trends --last 10
uv run raki trends --metrics first_pass_success_rate,rework_cycles
```

## Baseline thresholds

These thresholds are derived from v0.8.0 cycle data. Adjust them as your pipeline matures.

| Metric | v0.8.0 Baseline | Target | Gate Expression |
|--------|----------------|--------|----------------|
| `first_pass_success_rate` | 0.44 | ≥ 0.6 | `first_pass_success_rate>=0.6` |
| `rework_cycles` | 0.72 | ≤ 0.5 | `rework_cycles<=0.5` |
| `review_severity_distribution` | 0.63 | ≥ 0.5 | `review_severity_distribution>=0.5` |
| `cost_efficiency` | $7.28 avg | ≤ $10.0 | `cost_efficiency<=10.0` |

The built-in `SODA_DEFAULT_GATES` (the two most critical gates applied when no overrides are given):

```
first_pass_success_rate>=0.6
rework_cycles<=0.5
```

## Customizing thresholds

### Via `--gate` flag

Override defaults with per-invocation gates. CLI `--gate` flags replace SODA defaults entirely:

```bash
# Stricter first-pass target
uv run raki gate-check results/current.json \
  --gate 'first_pass_success_rate>=0.75' \
  --gate 'rework_cycles<=0.3'
```

### Via manifest `thresholds:` block

Use the provided manifest template for repeatable gate configuration:

```bash
# Use soda-gate.yaml thresholds (overrides SODA defaults)
uv run raki gate-check results/current.json -m examples/soda-gate.yaml
```

Priority chain (highest to lowest): CLI `--gate` > manifest `thresholds:` > SODA defaults.

### Manifest template

`examples/soda-gate.yaml` is a ready-to-use manifest with all four SODA metrics pre-configured.
Copy and adjust it for your team's targets:

```yaml
sessions:
  path: .soda/
  format: auto

thresholds:
  - "first_pass_success_rate>=0.6"
  - "rework_cycles<=0.5"
  - "review_severity_distribution>=0.5"
  - "cost_efficiency<=10.0"
```

## CI integration

Add a milestone gate check to your GitHub Actions workflow:

```yaml
name: SODA Milestone Quality Gate

on:
  workflow_dispatch:
    inputs:
      current_report:
        description: "Path to the current milestone report JSON"
        required: true
      baseline_report:
        description: "Path to the baseline (previous milestone) report JSON"
        required: false

jobs:
  gate-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"
      - name: Install RAKI
        run: uv sync --python 3.12 --all-extras
      - name: Run SODA sessions evaluation
        run: |
          uv run raki run -m examples/soda-gate.yaml -o results/
      - name: Check quality gates
        run: |
          uv run raki gate-check results/raki-report-*.json \
            -m examples/soda-gate.yaml
      - name: Check regression vs baseline
        if: inputs.baseline_report != ''
        run: |
          uv run raki gate-check results/raki-report-*.json \
            --baseline ${{ inputs.baseline_report }} \
            -m examples/soda-gate.yaml
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: soda-quality-report
          path: results/
          retention-days: 90
```

## Interpreting results

### When gates fail

**`first_pass_success_rate` below threshold:**
- Review recent changes to the `implement` phase prompt — did the instruction change in a way that reduces clarity?
- Check if ticket complexity increased (larger ticket budget, more files, more unknowns).
- Use `raki report --diff` to see which sessions are transitioning from pass to rework.

**`rework_cycles` above threshold:**
- Examine the `verify` prompt feedback messages — are they actionable?
- Check if the `implement` agent is ignoring verify feedback (common in short-budget sessions).
- Look for patterns: same ticket type, same phase, same error class.

**`review_severity_distribution` below threshold:**
- More critical/major findings means review quality may be improving (catching more issues) or implementation quality dropped.
- Correlate with `rework_cycles` — if both worsen together, implementation quality declined.

**`cost_efficiency` above threshold:**
- Check for runaway sessions (very high individual costs).
- Look at whether LLM provider or model changed between milestones.

### Using `--diff` to pinpoint regressions

After detecting a regression with `gate-check`, use `raki report --diff` for per-session detail:

```bash
uv run raki report \
  --diff results/previous.json results/current.json \
  --fail-on-regression
```

This shows which sessions changed verdict (pass → rework, rework → pass) between milestones.

### Using `raki trends` for trajectory analysis

For multi-milestone trend analysis:

```bash
# Show all metric trends for the last 10 evaluations
uv run raki trends --last 10

# Focus on the two most critical SODA metrics
uv run raki trends --metrics first_pass_success_rate,rework_cycles
```

A consistently flat or declining `first_pass_success_rate` with rising `rework_cycles` signals that prompt changes are not improving pipeline quality.
