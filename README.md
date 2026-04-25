# RAKI -- Retrieval Assessment for Knowledge Impact

A CLI tool that evaluates agentic RAG quality from session transcripts.

## Report Preview

![RAKI HTML Report](docs/images/report-screenshot.png)

## Three tiers of metrics

| Tier | What you need | Metrics |
|------|--------------|---------|
| **Operational** | Nothing (zero config) | Verify rate, rework cycles, cost, severity, latency, tokens, self-correction |
| **Knowledge** | `--docs-path` | Knowledge gap rate, knowledge miss rate |
| **Analytical** | `--judge` | Faithfulness, answer relevancy, context precision, context recall |

## Features

- **Operational metrics** -- verify rate, rework cycles, severity, cost, latency, tokens, self-correction (no LLM required)
- **Knowledge metrics** -- gap rate and miss rate based on project documentation coverage
- **Analytical metrics** -- Ragas-backed context precision/recall, faithfulness, answer relevancy (LLM judge)
- **HTML reports** -- interactive reports with session-level detail and color-coded thresholds
- **Quality gates** -- per-metric `--gate` thresholds and `--fail-on-regression` for CI
- **Pluggable adapters** -- bring any session format; built-in support for session-schema and Alcove

## Quick Start

```bash
# Install
uv pip install raki[html]

# Validate manifest
uv run raki validate --manifest raki.yaml

# Run operational metrics (default, no API keys needed)
uv run raki run --manifest raki.yaml

# Add knowledge metrics
uv run raki run --manifest raki.yaml --docs-path ./docs

# Add analytical metrics (requires LLM credentials)
uv run raki run --manifest raki.yaml --judge --judge-provider anthropic

# Add analytical metrics via LiteLLM (e.g. OpenAI)
uv run raki run --manifest raki.yaml --judge --judge-provider litellm --judge-model gpt-4o
```

## Usage

```bash
# Run all tiers (operational + knowledge + analytical)
uv run raki run --manifest raki.yaml --docs-path ./docs --judge

# Run with direct Anthropic API
uv run raki run --manifest raki.yaml --judge --judge-provider anthropic

# Run with LiteLLM (any LiteLLM-supported model)
uv run raki run --manifest raki.yaml --judge --judge-provider litellm --judge-model gpt-4o

# Run specific metrics only
uv run raki run --manifest raki.yaml --metrics cost_efficiency,rework_cycles

# Quality gates for CI
uv run raki run --manifest raki.yaml \
  --gate 'first_pass_verify_rate>0.85' \
  --gate 'rework_cycles<1.5' \
  --quiet

# List available metrics
uv run raki metrics

# Validate manifest and session data
uv run raki validate --manifest raki.yaml --deep

# Compare two evaluation runs
uv run raki report --diff results/baseline.json results/compare.json --fail-on-regression

# List available adapters
uv run raki adapters
```

## Documentation

- [Getting Started](docs/getting-started.md) -- install, run, and understand your first report
- **Metric references:**
  - [Operational Metrics](docs/metrics/operational.md) -- verify rate, rework, cost, severity, latency, tokens, self-correction
  - [Knowledge Metrics](docs/metrics/knowledge.md) -- gap rate, miss rate, domain matching
  - [Analytical Metrics](docs/metrics/analytical.md) -- faithfulness, relevancy, precision, recall
- [CI Integration Guide](docs/ci-integration.md) -- quality gates, regression detection, GitHub Actions / GitLab CI
- [Results Interpretation Reference](docs/interpretation-reference.md) -- zone tables and common patterns
- [Ground Truth Curation Guide](docs/curation-guide.md) -- write effective ground truth entries
- [Adapter Guide](docs/adapter-guide.md) -- integrate custom session formats
- [Session Schema Reference](docs/session-schema.md) -- field definitions for session-schema format

## Development

```bash
uv sync --python 3.12 --all-extras
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run ty check src/raki/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow.

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
