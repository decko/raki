# RAKI — Retrieval Assessment for Knowledge Impact

A CLI tool that evaluates agentic RAG quality from session transcripts.

## Report Preview

![RAKI HTML Report](docs/images/report-screenshot.png)

## Features

- **Operational metrics** — verify rate, rework cycles, severity distribution, cost analysis (no LLM required)
- **Ragas retrieval metrics** — context precision/recall, faithfulness, answer relevancy
- **HTML reports** — interactive reports with session-level detail and color-coded thresholds
- **Pluggable adapters** — bring any session format; built-in support for session-schema and Alcove
- **Ground truth matching** — curated YAML entries matched by domain for retrieval evaluation

## Quick Start

```bash
uv pip install raki --extra html
uv run raki validate --manifest raki.yaml
uv run raki run --manifest raki.yaml --no-llm
```

The `--no-llm` flag runs operational metrics only (no API keys needed).
To include Ragas retrieval metrics, omit the flag and configure an LLM provider.

## Usage

```bash
# Run all metrics (requires LLM provider)
uv run raki run --manifest raki.yaml

# Run operational metrics only
uv run raki run --manifest raki.yaml --no-llm

# Run specific metrics only
uv run raki run --manifest raki.yaml --no-llm --metrics cost_efficiency,rework_cycles

# Validate manifest and session data
uv run raki validate --manifest raki.yaml

# List available adapters
uv run raki adapters

# List available metrics (name, display name, LLM requirement)
uv run raki metrics

# Machine-readable metric list
uv run raki metrics --json
```

## Documentation

- [Getting Started](docs/getting-started.md) — install, run, and understand your first report
- [Interpretation Reference](docs/interpretation-reference.md) — what each metric means and when to act
- [Ground Truth Curation Guide](docs/curation-guide.md) — write effective ground truth entries
- [Adapter Guide](docs/adapter-guide.md) — integrate custom session formats
- [Session Schema Reference](docs/session-schema.md) — field definitions for session-schema format

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

Apache 2.0 — see [LICENSE](LICENSE) for details.
