# CI Integration Guide

RAKI provides quality gates, regression detection, and structured exit codes for CI pipelines.

## Quality gates with --gate

The `--gate` flag sets per-metric thresholds that control the exit code. If any gate fails, RAKI exits with code 1.

### Syntax

```
--gate 'metric_name>value'
--gate 'metric_name>=value'
--gate 'metric_name<value'
--gate 'metric_name<=value'
```

Supported operators: `>`, `<`, `>=`, `<=`.

### Examples

```bash
# Require verify rate above 85%
uv run raki run --manifest raki.yaml --gate 'first_pass_success_rate>0.85'

# Multiple gates
uv run raki run --manifest raki.yaml \
  --gate 'first_pass_success_rate>0.85' \
  --gate 'rework_cycles<1.5' \
  --gate 'cost_efficiency<15.0'

# With analytical metrics
uv run raki run --manifest raki.yaml --judge \
  --gate 'faithfulness>0.80' \
  --gate 'context_precision>0.75'
```

### Manifest-based thresholds

You can also define thresholds in your manifest file (`raki.yaml`). CLI `--gate` flags override manifest thresholds when both are present.

```yaml
thresholds:
  - "first_pass_success_rate>0.85"
  - "rework_cycles<1.5"
```

## Requiring metrics with --require-metric

By default, when a metric is N/A (no applicable data), its gate is skipped and treated as passing. Use `--require-metric` to fail instead:

```bash
uv run raki run --manifest raki.yaml \
  --gate 'self_correction_rate>0.80' \
  --require-metric self_correction_rate
```

If `self_correction_rate` is N/A (no rework findings exist), this fails the gate instead of silently skipping it.

## Regression detection with --fail-on-regression

Compare two evaluation runs and fail if any metric regressed beyond a noise margin (2%):

```bash
uv run raki report \
  --diff results/baseline.json results/current.json \
  --fail-on-regression
```

Regression direction respects each metric's `higher_is_better` setting. For example, an increase in `rework_cycles` (lower is better) is flagged as a regression.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All gates passed, no regressions |
| 1 | Quality gate violation (`--gate` threshold failed) |
| 2 | Configuration or input error (bad manifest, unknown metric, invalid syntax) |
| 3 | Regression detected (`--fail-on-regression`) |
| 4 | Both threshold violation and regression detected |

## GitHub Actions example

```yaml
name: RAG Quality Gate

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 3 * * *"
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"
      - name: Install RAKI
        run: uv sync --python 3.12 --all-extras
      - name: Validate manifest
        run: uv run raki validate --manifest raki.yaml

  operational-gate:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"
      - name: Install RAKI
        run: uv sync --python 3.12 --all-extras
      - name: Run operational metrics with gates
        run: |
          uv run raki run \
            --manifest raki.yaml \
            --gate 'first_pass_success_rate>0.85' \
            --gate 'rework_cycles<1.5' \
            --output results/ \
            --quiet
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: raki-operational-report
          path: results/
          retention-days: 30

  nightly-judge-gate:
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    needs: validate
    runs-on: ubuntu-latest
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"
      - name: Install RAKI
        run: uv sync --python 3.12 --all-extras
      - name: Run full evaluation with judge
        run: |
          uv run raki run \
            --manifest raki.yaml \
            --judge \
            --judge-provider anthropic \
            --gate 'faithfulness>0.80' \
            --gate 'first_pass_success_rate>0.85' \
            --include-sessions \
            --output results/ \
            --quiet
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: raki-full-report
          path: results/
          retention-days: 90
```

## GitLab CI example

```yaml
stages:
  - validate
  - gate

variables:
  UV_VERSION: "latest"

.uv-setup: &uv-setup
  before_script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - export PATH="$HOME/.local/bin:$PATH"
    - uv sync --python 3.12 --all-extras

validate:
  stage: validate
  <<: *uv-setup
  script:
    - uv run raki validate --manifest raki.yaml

operational-gate:
  stage: gate
  <<: *uv-setup
  needs: [validate]
  script:
    - |
      uv run raki run \
        --manifest raki.yaml \
        --gate 'first_pass_success_rate>0.85' \
        --gate 'rework_cycles<1.5' \
        --output results/ \
        --quiet
  artifacts:
    paths:
      - results/
    expire_in: 30 days
    when: always

nightly-judge-gate:
  stage: gate
  <<: *uv-setup
  needs: [validate]
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
  variables:
    ANTHROPIC_API_KEY: $ANTHROPIC_API_KEY
  script:
    - |
      uv run raki run \
        --manifest raki.yaml \
        --judge \
        --judge-provider anthropic \
        --gate 'faithfulness>0.80' \
        --include-sessions \
        --output results/ \
        --quiet
  artifacts:
    paths:
      - results/
    expire_in: 90 days
    when: always
```

## SODA pipeline quality gate

Use `raki gate-check` to evaluate SODA pipeline sessions on a per-milestone cadence.
Unlike `raki run`, which evaluates sessions and computes metrics, `gate-check` operates
on an already-generated JSON report and applies SODA-tuned thresholds in one fast step.

Default SODA gates (applied automatically with no additional flags):
- `first_pass_success_rate>=0.6` — target to improve from the v0.8.0 baseline of 0.44
- `rework_cycles<=0.5` — target to reduce from the v0.8.0 baseline of 0.72

### Quick start

```bash
# Step 1: Run evaluation on SODA sessions
uv run raki run -m examples/soda-gate.yaml -o results/

# Step 2: Check gates with SODA defaults
uv run raki gate-check results/raki-report-*.json

# Step 3: Compare against a previous milestone baseline
uv run raki gate-check results/current.json --baseline results/previous.json

# Step 4: Use custom thresholds or a manifest
uv run raki gate-check results/current.json -m examples/soda-gate.yaml
```

### GitHub Actions snippet for milestone gate checks

```yaml
name: SODA Milestone Quality Gate

on:
  workflow_dispatch:

jobs:
  soda-gate-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          version: "latest"
      - name: Install RAKI
        run: uv sync --python 3.12 --all-extras
      - name: Run SODA session evaluation
        run: |
          uv run raki run \
            --manifest examples/soda-gate.yaml \
            --output results/ \
            --quiet
      - name: Check SODA quality gates
        run: |
          uv run raki gate-check results/raki-report-*.json \
            --manifest examples/soda-gate.yaml
      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: soda-quality-report
          path: results/
          retention-days: 90
```

See [docs/soda-pipeline-gate.md](soda-pipeline-gate.md) for the full per-milestone workflow,
baseline threshold table, and interpretation guide.

## Tips

- **Start with operational gates only.** They run fast, need no API keys, and catch the most common issues. Add analytical gates after you have stable baselines.
- **Use `--quiet` in CI.** Suppresses progress output and only prints errors and gate results.
- **Use `--json` for machine parsing.** `uv run raki run --manifest raki.yaml --json` writes the full JSON report to stdout.
- **Archive reports as artifacts.** JSON reports can be re-rendered later with `raki report` or compared with `raki report --diff`.
- **Regression detection across PRs.** Store the baseline JSON from your main branch, then use `--fail-on-regression` to compare against it on each PR.
