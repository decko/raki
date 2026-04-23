# CI Workflow Guide — Agent Quality Gates with RAKI

This guide shows how to set up a CI pipeline that catches agent quality regressions before they reach production. The key idea: run your agent against **known test tickets**, evaluate the results with RAKI, and block merges when quality drops.

## Prerequisites

You need:
1. A set of **test tickets** with known expected outcomes
2. Session transcripts from running your agent against those tickets
3. A RAKI manifest pointing at those sessions
4. A **baseline** — the evaluation results from your last known-good run

## Step 1: Create Test Tickets

Pick 5-10 tickets that represent your agent's typical workload. Include a mix:

| Ticket | Complexity | Expected Outcome |
|--------|-----------|-----------------|
| PROJ-101 | small | Pass on first try, no rework |
| PROJ-102 | medium | Pass with minor findings |
| PROJ-103 | medium | May need 1 rework cycle |
| PROJ-104 | large | Complex, tests the agent's limits |
| PROJ-105 | small (bug fix) | Should be quick and clean |

These become your **regression test suite for agent behavior**. They don't change — when you update prompts, knowledge base, or model, you re-run the agent against these same tickets and compare.

## Step 2: Generate Baseline Sessions

Run your agent against the test tickets and save the session transcripts:

```bash
# Run your orchestrator/soda pipeline against each test ticket
soda run PROJ-101
soda run PROJ-102
soda run PROJ-103
soda run PROJ-104
soda run PROJ-105

# Session transcripts are saved in .soda/ (soda) or your session directory
```

The session format depends on your orchestrator:
- **Soda**: sessions in `.soda/<ticket>/` (meta.json + events.jsonl + phase files)
- **Claude Code / Alcove**: single JSON transcript files
- **Bridge**: JSON files with `id` + `task_id` + `transcript`

## Step 3: Create RAKI Manifest

Point RAKI at your test sessions:

```yaml
# raki-ci.yaml
sessions:
  path: .ci/test-sessions/    # curated test session transcripts
  format: auto                # auto-detect soda or alcove format

# Optional: point at your project docs for knowledge metrics
docs:
  path: docs/
  extensions: [.md, .txt]
```

## Step 4: Establish the Baseline

Run RAKI and save the result as your baseline:

```bash
# Operational metrics (no LLM needed, fast, free)
raki run --manifest raki-ci.yaml --output .ci/results/

# Or with full analytical metrics (needs LLM credentials)
raki run --manifest raki-ci.yaml --judge --docs-path docs/ --output .ci/results/
```

Review the output. If the numbers look right, save as baseline:

```bash
cp .ci/results/raki-report-*.json .ci/baseline.json
```

Commit the baseline and test sessions to your repo:
```
.ci/
  raki-ci.yaml
  baseline.json
  test-sessions/
    PROJ-101/          # soda format
      meta.json
      events.jsonl
      triage.json
      implement.json
      ...
    PROJ-102/
      ...
```

## Step 5: CI Pipeline

### On Every PR — Operational Gate (fast, free)

Run RAKI with absolute thresholds. No LLM needed — evaluates cost, rework, success rate from session data alone:

```yaml
# .github/workflows/agent-quality.yml
name: Agent Quality Gate

on:
  pull_request:
    branches: [main]

jobs:
  agent-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install RAKI
        run: pip install raki

      - name: Run operational quality gate
        run: |
          raki run \
            --manifest .ci/raki-ci.yaml \
            --gate 'first_pass_success_rate>0.40' \
            --gate 'rework_cycles<2.0' \
            --gate 'cost_efficiency<20.0'

      - name: Compare against baseline
        run: |
          raki report \
            --diff .ci/baseline.json results/raki-report-*.json \
            --fail-on-regression
```

### On Merge — Update Baseline

When a PR merges, the successful evaluation becomes the new baseline:

```yaml
  update-baseline:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install RAKI
        run: pip install raki

      - name: Re-run evaluation
        run: raki run --manifest .ci/raki-ci.yaml --output .ci/results/

      - name: Update baseline
        run: |
          cp .ci/results/raki-report-*.json .ci/baseline.json
          git config user.name "CI Bot"
          git config user.email "ci@example.com"
          git add .ci/baseline.json
          git commit -m "chore: update raki baseline"
          git push
```

### Nightly — Full Analytical Gate (with LLM)

LLM-judged metrics cost money, so run them on a schedule rather than every PR:

```yaml
  nightly-judge:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    env:
      GOOGLE_CLOUD_PROJECT: ${{ secrets.GCP_PROJECT }}
      VERTEXAI_LOCATION: us-east5
    steps:
      - uses: actions/checkout@v4

      - name: Install RAKI
        run: pip install 'raki[ragas]'

      - name: Full evaluation with judge
        run: |
          raki run \
            --manifest .ci/raki-ci.yaml \
            --judge \
            --docs-path docs/ \
            --gate 'faithfulness>0.20' \
            --gate 'first_pass_success_rate>0.40' \
            --include-sessions

      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: raki-nightly-report
          path: results/
```

## Step 6: Re-generate Sessions After Changes

When you change your agent's prompts, knowledge base, or model:

1. **Re-run the agent** against the same test tickets
2. **Copy the new sessions** into `.ci/test-sessions/`
3. **Run RAKI** — the `--diff` against baseline shows what changed
4. **Review the diff** — did your changes improve or regress quality?
5. **Update the baseline** if the new results are the expected new normal

```bash
# After changing your agent's prompts
soda run PROJ-101
soda run PROJ-102
# ... run all test tickets

# Copy new sessions
cp -r .soda/PROJ-101 .ci/test-sessions/
cp -r .soda/PROJ-102 .ci/test-sessions/
# ...

# Evaluate and compare
raki run --manifest .ci/raki-ci.yaml --output .ci/results/
raki report --diff .ci/baseline.json .ci/results/raki-report-*.json

# If improvements look good, update baseline
cp .ci/results/raki-report-*.json .ci/baseline.json
```

## Exit Codes Reference

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All gates pass, no regressions | Merge is safe |
| 1 | Threshold violation | A metric is below the quality bar |
| 2 | Input error | Bad manifest, invalid gate syntax |
| 3 | Regression detected | A metric got worse vs baseline |
| 4 | Both threshold and regression | Quality dropped and is below the bar |

## What the Metrics Tell You

| Change | Signal | Action |
|--------|--------|--------|
| Prompt update → success rate drops | Agent gets confused by new instructions | Simplify the prompt, add examples |
| KB update → knowledge gap drops | New docs cover previously uncovered domains | Working as intended |
| Model change → faithfulness drops | Cheaper model hallucinates more | May not be worth the cost saving |
| Model change → cost drops, quality stable | More efficient model | Good trade-off |
| Rework cycles increase | Review is catching more issues | Check if findings are legitimate or review is too strict |

## Using `--diff` Beyond CI

The diff command compares any two RAKI evaluation runs. Beyond CI regression detection, here are practical ways to use it:

### Prompt A/B Testing

Run the same test tickets with two different system prompts and compare:

```bash
# Run with prompt A
raki run --manifest raki-ci.yaml --output results-a/

# Switch to prompt B, re-run the agent on the same tickets
raki run --manifest raki-ci.yaml --output results-b/

# Compare
raki report --diff results-a/raki-report-*.json results-b/raki-report-*.json
```

Look for: did faithfulness improve without hurting success rate? Did rework cycles drop? If prompt B is better on analytical metrics but worse on operational, it might be producing prettier text but not better code.

### Model Comparison

Evaluate the same tickets with different models to make cost/quality trade-offs:

```bash
# Run with Sonnet (cheaper)
soda run PROJ-101 --model claude-sonnet-4-6
raki run --manifest raki-ci.yaml --output results-sonnet/

# Run with Opus (more expensive)
soda run PROJ-101 --model claude-opus-4-6
raki run --manifest raki-ci.yaml --output results-opus/

# Compare
raki report --diff results-sonnet/raki-report-*.json results-opus/raki-report-*.json
```

Look for: does Opus produce fewer rework cycles? Is the cost increase justified by quality gains? If Sonnet has `rework_cycles=1.5` at `$5/session` and Opus has `rework_cycles=0.3` at `$15/session`, the total cost (including rework) might favor Opus.

### Knowledge Base Impact

Measure the before/after effect of adding documentation:

```bash
# Baseline: run without docs
raki run --manifest raki-ci.yaml --output results-no-docs/

# Add your new runbook/AGENTS.md/gotchas doc
raki run --manifest raki-ci.yaml --docs-path docs/ --output results-with-docs/

# Compare
raki report --diff results-no-docs/raki-report-*.json results-with-docs/raki-report-*.json
```

Look for: did `knowledge_gap_rate` drop? That means your new docs cover domains where the agent was previously failing without guidance. If `knowledge_miss_rate` is still high, the docs exist but the agent isn't applying them — check retrieval or prompt injection of context.

### Weekly Trend Tracking

Run nightly evaluations and diff each week against the previous:

```bash
# Cron job saves weekly snapshots
raki run --manifest raki-ci.yaml --judge --output "results/week-$(date +%V)/"

# Compare this week vs last week
raki report --diff results/week-15/raki-report-*.json results/week-16/raki-report-*.json
```

Look for: gradual drift. Faithfulness might drop 2% per week as the codebase evolves and the knowledge base falls behind. A single week's drop is noise — three weeks of consistent decline is a signal to update your docs.

### Post-Incident Analysis

After an agent produces a bad outcome in production, run raki on the session and diff against your healthy baseline:

```bash
# Evaluate the problematic session
raki run --manifest incident-manifest.yaml --judge --output results-incident/

# Compare against known-good baseline
raki report --diff .ci/baseline.json results-incident/raki-report-*.json
```

Look for: which metrics diverged? If `faithfulness` dropped, the agent hallucinated. If `knowledge_gap_rate` spiked, the agent worked in a domain with no documentation. If `self_correction_rate` dropped, the agent couldn't recover from review feedback. The diff tells you where the failure originated, not just that it happened.
