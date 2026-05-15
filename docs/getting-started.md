# Getting Started with RAKI

Evaluate your agentic RAG sessions in under 10 minutes.

## Prerequisites

- **Python 3.12** or later
- **[uv](https://docs.astral.sh/uv/)** package manager

## Install

```bash
uv pip install raki
```

For HTML reports, install the `html` extra:

```bash
uv pip install raki[html]
```

For development or analytical metrics (Ragas), install all extras:

```bash
git clone https://github.com/decko/raki.git
cd raki
uv sync --python 3.12 --all-extras
```

> **Note:** The `ragas` extra pulls `scikit-network`, which requires a C++ compiler (`g++`).

Verify the install:

```bash
uv run raki --help
```

## Three tiers of metrics

RAKI organizes metrics into three tiers, each adding more depth:

| Tier | What you need | What you get |
|------|--------------|--------------|
| **Operational** | Nothing (zero config) | First-pass success rate, rework cycles, cost, severity, latency, tokens, self-correction |
| **Knowledge** | `--docs-path ./docs` | Knowledge gap rate, knowledge miss rate |
| **Analytical** | `--judge` + LLM credentials | Faithfulness, answer relevancy, context precision, context recall |

Start with operational metrics and add tiers as needed.

## Step 1: Operational metrics (zero config)

Run operational metrics immediately -- no API keys, no docs, no ground truth:

```bash
uv run raki run --manifest raki.yaml
```

This is the default mode. It computes all seven operational metrics from session transcript data alone:

- **First-pass success rate** -- % sessions with no rework cycles
- **Rework cycles** -- mean review-fix iterations per session
- **Severity score** -- weighted severity of review findings
- **Cost / session** -- mean USD cost per session
- **Self-correction rate** -- ratio of rework findings resolved
- **Phase execution time** -- mean phase time in seconds
- **Tokens / phase** -- mean tokens per phase

### Discovering metrics

```bash
uv run raki metrics          # table of all metrics
uv run raki metrics --json   # machine-readable
```

### Running specific metrics

```bash
uv run raki run --manifest raki.yaml --metrics cost_efficiency,rework_cycles
```

## Step 2: Add docs for knowledge metrics

Point RAKI at your project documentation to activate knowledge metrics:

```bash
uv run raki run --manifest raki.yaml --docs-path ./docs
```

Or configure it in your manifest:

```yaml
docs:
  path: ./docs
  extensions: [".md", ".rst", ".txt"]
```

This adds two metrics:

- **Knowledge gap rate** -- how often rework happens in domains not covered by your docs
- **Knowledge miss rate** -- how often the agent fails despite having relevant docs

See [Knowledge Metrics Reference](metrics/knowledge.md) for details.

## Step 3: Add a judge for analytical metrics

Enable LLM-judged retrieval quality metrics with `--judge`:

```bash
# Vertex AI Anthropic (default)
uv run raki run --manifest raki.yaml --judge

# Direct Anthropic API
uv run raki run --manifest raki.yaml --judge --judge-provider anthropic

# Google AI
uv run raki run --manifest raki.yaml --judge --judge-provider google

# LiteLLM (any model via the LiteLLM proxy, e.g. OpenAI)
uv run raki run --manifest raki.yaml --judge \
  --judge-provider litellm --judge-model gpt-4o
```

This adds four Ragas-backed metrics:

- **Faithfulness** -- is the output grounded in retrieved context?
- **Answer relevancy** -- does the output address the question?
- **Context precision** -- is the retrieved context relevant? (requires ground truth)
- **Context recall** -- was all needed context retrieved? (requires ground truth)

Set `ANTHROPIC_API_KEY` for direct Anthropic API, or configure Google Cloud credentials for Vertex AI.
For LiteLLM, set the appropriate provider credentials (e.g. `OPENAI_API_KEY`) and install the extra:

```bash
uv pip install raki[litellm]
```

See [Analytical Metrics Reference](metrics/analytical.md) for details.

### Persisting judge config in your manifest

Instead of passing `--judge-provider` and `--judge-model` on every run, you can
save them in your manifest:

```yaml
judge:
  provider: vertex-anthropic
  model: claude-sonnet-4-6
```

The `--judge` flag is still required to enable analytical metrics — the manifest
just persists *which* provider and model to use.

### Judge config resolution order

RAKI resolves judge provider and model using a 4-tier priority chain:

1. **CLI flags** (`--judge-provider`, `--judge-model`) — highest priority
2. **Manifest** (`judge.provider`, `judge.model`)
3. **Environment variables** (`RAKI_JUDGE_PROVIDER`, `RAKI_JUDGE_MODEL`)
4. **Built-in defaults** (`vertex-anthropic`, `claude-sonnet-4-6`)

This means you can set defaults in your manifest, override per-environment via
env vars, and still override everything on the command line.

## Validate before running

Check your manifest and session data without running metrics:

```bash
uv run raki validate --manifest raki.yaml
```

For a deeper smoke test (adapter loading, ground truth wiring, metric computation against one sample):

```bash
uv run raki validate --manifest raki.yaml --deep
```

## Understanding the output

A typical operational run produces:

```
Operational Health
  First-pass success rate              0.75
  Rework cycles                       0.2
  Severity score                      0.39
  Cost / session                      $10.93
  Self-correction rate                N/A (no rework findings)
  Phase execution time                142.3s
  Tokens / phase                      3,241
```

Reports are saved as JSON (always) and HTML (when jinja2 is installed). Re-render anytime:

```bash
uv run raki report results/raki-report-20260410T120000.json
uv run raki report results/raki-report-20260410T120000.json --html report.html
```

To compare two runs and see metric deltas, direction indicators, and per-session verdict transitions, use the `--diff` subcommand:

```bash
uv run raki report --diff results/before.json results/after.json
```

For the full before/after comparison workflow — including how to scope manifests, what the diff output shows, and how to gate CI on regressions — see [Comparing Runs](comparing-runs.md).

## CI integration

Use `--gate` for per-metric quality gates:

```bash
uv run raki run --manifest raki.yaml \
  --gate 'first_pass_success_rate>0.85' \
  --gate 'rework_cycles<1.5' \
  --quiet
```

See [CI Integration Guide](ci-integration.md) for `--gate` syntax, `--fail-on-regression`, exit codes, and full GitHub Actions / GitLab CI examples.

## Next steps

- [Operational Metrics Reference](metrics/operational.md) -- all seven operational metrics in detail
- [Knowledge Metrics Reference](metrics/knowledge.md) -- knowledge gap and miss rates
- [Analytical Metrics Reference](metrics/analytical.md) -- Ragas-backed retrieval quality metrics
- [CI Integration Guide](ci-integration.md) -- quality gates, regression detection, CI examples
- [Results Interpretation Reference](interpretation-reference.md) -- zone tables and common patterns
- [Ground Truth Curation Guide](curation-guide.md) -- writing ground truth for context precision/recall
- [Adapter Guide](adapter-guide.md) -- integrating custom session formats
