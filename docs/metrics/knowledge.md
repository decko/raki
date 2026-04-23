# Knowledge Metrics Reference

Knowledge metrics measure how well your project's documentation covers the domains where the agent makes mistakes. They require a docs path (via `--docs-path` or the `docs.path` manifest field) and activate automatically when docs are loaded.

> **See also:** [Rationale and Interpretation Guide](rationale-and-interpretation.md) — detailed design rationale, interpretation tables, pitfall warnings, and combined metric patterns for all non-Ragas metrics.

## Prerequisites

Provide project documentation so RAKI can build a knowledge context:

```bash
uv run raki run --manifest raki.yaml --docs-path ./docs
```

Or configure it in your manifest:

```yaml
docs:
  path: ./docs
  extensions: [".md", ".rst", ".txt"]
```

## How domain matching works

RAKI loads doc files from the docs path, chunks them, and derives **domains** from the directory structure (e.g., `docs/api/auth.md` maps to the `api` domain). Each chunk carries its domain tag.

When computing knowledge metrics, RAKI compares the words in each review finding's `issue` text against the words in the knowledge context. A finding is considered **covered** if its issue words overlap with the knowledge context, and **uncovered** if they do not.

- **Covered domains:** The knowledge base has content relevant to the finding's topic.
- **Uncovered domains:** No knowledge base content matches the finding's topic.

## knowledge_gap_rate -- Knowledge gap rate

**What it measures:** Ratio of rework-triggering findings (critical/major) in domains NOT covered by the knowledge base.

**What it tells you:** Where your documentation is missing. High values mean agents are making mistakes in areas where no reference material exists.

**What action it drives:** Extract the topics from uncovered findings and add them to your knowledge base. This is the single most direct way to improve agent quality.

**How it's computed:** For sessions with `rework_cycles > 0` and knowledge context: count findings whose issue words do NOT overlap with the knowledge context. Score = `uncovered_findings / total_rework_findings`. Lower is better.

**N/A conditions:** Returns `score=None` when:
- No sessions have rework findings, OR
- No sessions have knowledge context (docs not loaded)

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.20 | KB covers most failure domains |
| Yellow | 0.20--0.40 | Notable gaps in knowledge coverage |
| Red | > 0.40 | KB is missing content for many failure modes |

---

## knowledge_miss_rate -- Knowledge miss rate

**What it measures:** Ratio of rework-triggering findings (critical/major) in domains that ARE covered by the knowledge base but the agent still got wrong.

**What it tells you:** How often the agent fails despite having the right reference material. High values may indicate the KB content is unclear, outdated, or the agent is not using it effectively.

**What action it drives:** Review the KB content for the affected domains. The information exists but is not preventing mistakes -- it may need to be restructured, made more explicit, or moved closer to the agent's context window.

**How it's computed:** For sessions with `rework_cycles > 0` and knowledge context: count findings whose issue words overlap with the knowledge context. Score = `covered_findings / total_rework_findings`. Lower is better.

**N/A conditions:** Returns `score=None` when:
- No sessions have rework findings, OR
- No sessions have knowledge context (docs not loaded)

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.10 | Agent uses KB content effectively |
| Yellow | 0.10--0.30 | Agent sometimes ignores available knowledge |
| Red | > 0.30 | Agent frequently fails despite having KB coverage |

## Relationship between gap rate and miss rate

These two metrics are complementary:

- `knowledge_gap_rate + knowledge_miss_rate` may not sum to 1.0 because minor findings are excluded and sessions without knowledge context are skipped entirely.
- **High gap rate, low miss rate:** Your KB works well where it exists -- expand its coverage.
- **Low gap rate, high miss rate:** Your KB covers the right domains but the content is not effective -- improve quality.
- **Both high:** Both coverage and quality need work.
