# RAKI Development Pipeline (SODA)

This directory contains the soda pipeline configuration for developing RAKI itself. It defines how the autonomous agent triages, plans, implements, verifies, reviews, and submits changes.

## Pipeline Phases

```
triage → plan → implement → verify → review → submit
                    ↑                    │
                    └── rework ──────────┘
```

| Phase | Model | Purpose |
|-------|-------|---------|
| triage | Sonnet | Classify ticket, identify files, assess complexity |
| plan | Opus 1M | Design TDD implementation plan |
| implement | Sonnet | Write code following the plan |
| verify | Sonnet | Run tests, check acceptance criteria |
| review | Opus 1M | Multi-specialist code review (Python + Security + RAG + Doc) |
| submit | Sonnet | Create PR on GitHub |

## Setup

1. Copy the example config to your raki repo root:
   ```bash
   cp docs/soda/soda.example.yaml soda.yaml
   ```

2. Create symlinks for prompt/schema discovery:
   ```bash
   ln -sf docs/soda/prompts prompts
   ln -sf docs/soda/schemas schemas
   ```

3. Run on a ticket:
   ```bash
   soda run <issue-number>
   ```

## Files

- `phases.yaml` — phase definitions with tools, timeouts, retries, and model routing
- `soda.example.yaml` — project config template (copy to repo root as `soda.yaml`)
- `prompts/` — phase prompt templates with RAKI-specific conventions
- `schemas/` — JSON schemas for structured phase output
