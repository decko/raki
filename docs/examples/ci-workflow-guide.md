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
