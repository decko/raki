# Comparing Runs

Use `raki report --diff` to compare two evaluation runs side by side. This workflow answers the question "did this change improve things?" — quantified, with per-session detail.

## Quick start

This section walks through a complete before/after comparison in four commands.

### Step 1: Evaluate pre-change sessions

To evaluate the pre-change sessions, run the following command:

```bash
uv run raki run -m manifests/before.yaml -o results/ --include-sessions
```

RAKI writes `results/raki-report-<timestamp>.json`. Rename or copy it to a stable path so you can reference it later:

```bash
cp results/raki-report-*.json results/before.json
```

### Step 2: Evaluate post-change sessions

To evaluate the post-change sessions, run the following command:

```bash
uv run raki run -m manifests/after.yaml -o results/ --include-sessions
```

Copy the new report to a stable path:

```bash
cp results/raki-report-*.json results/after.json
```

### Step 3: Compare the two reports

To compare the two reports, run the following command:

```bash
uv run raki report --diff results/before.json results/after.json
```

Example output:

```
Comparing raki-20260501T100000 → raki-20260512T140000
Matched: 12/14 sessions (2 new)

  First-pass success rate  0.36 → 0.67  (+0.31)  ▲
  Rework cycles            0.72 → 0.33  (-0.39)  ▲
  Cost / session           $12.30 → $8.50  (-$3.80)  ▲
  Severity score           0.45 → 0.22  (-0.23)  ▲

Improvements: 4 sessions (fail→pass: 3, rework→pass: 1)
Regressions:  1 session (pass→rework: 1)
```

### Step 4: Generate an HTML diff report

To generate an HTML diff report, add `--html`:

```bash
uv run raki report --diff results/before.json results/after.json --html diff-report.html
```

The HTML report is a self-contained file with a dark theme. Open it in any browser.

### Step 5: Gate CI on regressions

To fail CI when metrics regress, add `--fail-on-regression`:

```bash
uv run raki report --diff results/baseline.json results/current.json --fail-on-regression
```

RAKI exits with code 3 when any metric regresses beyond the 2% noise margin. See [CI Integration](ci-integration.md) for exit code reference.

## Reading the diff output

### Header

The first line identifies the two runs being compared:

```
Comparing raki-20260501T100000 → raki-20260512T140000
```

The run IDs come from the report timestamps. The left-hand run is the baseline; the right-hand run is the comparison target.

### Judge config warnings

When the two reports used different judge providers or models, RAKI prints a warning before the metric table:

```
Warning: judge config differs between runs
  Baseline:  vertex-anthropic / claude-sonnet-4-6
  Compare:   anthropic / claude-opus-4
```

Analytical metric deltas (faithfulness, answer relevancy, context precision, context recall) may not be meaningful when the judge changed. Operational metric deltas are unaffected.

When only one report used a judge, RAKI warns that analytical metrics are missing from the other run and omits those metric rows from the diff.

### Agent model warnings

When the sessions in each report used different agent models, RAKI prints a warning noting that the model changed. This is informational — the diff proceeds, but metric differences may reflect model differences as much as prompt or config differences.

### Coverage line

The coverage line shows how many sessions were matched by `session_id`:

```
Matched: 12/14 sessions (2 new)
```

- **Matched**: sessions present in both reports, compared directly
- **New**: sessions present only in the comparison report
- **Dropped**: sessions present only in the baseline report

Aggregate metric deltas are computed from matched sessions only. New and dropped sessions are counted but do not affect the deltas.

### Metric deltas

Each metric is shown with its baseline value, comparison value, numeric delta, and direction indicator:

```
  First-pass success rate  0.36 → 0.67  (+0.31)  ▲
  Rework cycles            0.72 → 0.33  (-0.39)  ▲
  Cost / session           $12.30 → $8.50  (-$3.80)  ▲
  Severity score           0.45 → 0.22  (-0.23)  ▲
```

The direction indicator (`▲` / `▼` / `=`) respects each metric's `higher_is_better` setting:

| Indicator | Meaning |
|-----------|---------|
| `▲` | Improvement (value moved in the desired direction) |
| `▼` | Regression (value moved in the wrong direction) |
| `=` | No change |

For example, a decrease in `rework_cycles` is shown as `▲` because lower is better for that metric.

### Session transitions

When both reports were generated with `--include-sessions`, RAKI groups sessions by verdict change:

```
Improvements: 4 sessions (fail→pass: 3, rework→pass: 1)
Regressions:  1 session (pass→rework: 1)
```

Sessions with unchanged verdicts are not listed. Each transition line names the session ID and the verdict change so you can investigate specific sessions.

## Producing reports to compare

### Manifest scoping

Create one manifest per session set to scope each evaluation run to the correct sessions:

```yaml
# manifests/before.yaml — sessions from before the change
sessions:
  paths:
    - sessions/2026-04-01/
    - sessions/2026-04-15/
```

```yaml
# manifests/after.yaml — sessions from after the change
sessions:
  paths:
    - sessions/2026-05-01/
    - sessions/2026-05-12/
```

Alternatively, use a single manifest that covers all sessions and run it twice — once pointing at the before session directories and once at the after directories.

### The --include-sessions flag

Pass `--include-sessions` to both evaluation runs to enable per-session verdict transitions in the diff output. Without it, RAKI still shows aggregate metric deltas but cannot report which individual sessions improved or regressed.

To evaluate with per-session data, run the following command:

```bash
uv run raki run -m manifests/before.yaml -o results/ --include-sessions
```

### Consistent judge configuration

Verify that both reports use the same judge configuration (provider and model) when comparing analytical metrics. RAKI warns when they differ, but does not block the comparison. To ensure consistency, pin the judge config in both manifests:

```yaml
judge:
  provider: vertex-anthropic
  model: claude-sonnet-4-6
```

## When to use --diff vs raki trends

Both `raki report --diff` and `raki trends` help you understand whether evaluation quality is improving. They answer different questions.

| Situation | Use |
|-----------|-----|
| Before/after a specific prompt change | `raki report --diff` |
| Before/after a config or model change | `raki report --diff` |
| Monitoring quality over many runs | `raki trends` |
| Checking if a PR introduces a regression | `raki report --diff --fail-on-regression` |
| Weekly or monthly trajectory review | `raki trends` |
| Identifying a regression's first occurrence | `raki trends` |

**`raki report --diff`** is a point-in-time comparison between two specific evaluation runs. Use it when you have a concrete before/after hypothesis and want to confirm that a change had the intended effect.

**`raki trends`** shows metric trajectories over many runs. Use it for ongoing monitoring — for example, to verify that quality is improving sprint over sprint, or to spot a gradual degradation that no single diff would catch.

See the `raki trends --help` output for trajectory options such as `--since`, `--last`, and `--metrics`.

## Tips and caveats

- **Reports without `--include-sessions`** still show aggregate metric deltas. Only per-session verdict transitions (improvements and regressions) require session data in both reports.

- **New and dropped sessions** are displayed in the coverage line and counted separately. They do not affect aggregate metric deltas, which are computed from matched sessions only.

- **Judge config mismatches** trigger a warning but do not block comparison. Operational metric deltas remain meaningful regardless of judge config. Analytical metric deltas may reflect judge differences rather than true quality changes when the judge changed between runs.

- **Exit codes** from `raki report --diff`:

  | Code | Meaning |
  |------|---------|
  | 0 | No regressions detected |
  | 3 | Regression detected (`--fail-on-regression`) |
  | 4 | Regression and threshold violation |

  See [CI Integration](ci-integration.md) for the full exit code reference.

- **Noise margin**: `--fail-on-regression` uses a 2% noise margin. Metric changes smaller than 2% are treated as flat and do not trigger a regression exit.
