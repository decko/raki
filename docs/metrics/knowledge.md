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

---

## knowledge_gap_rate — Knowledge gap rate

**What it measures:** Ratio of rework-triggering findings (critical/major) in domains NOT covered by the knowledge base.

**What it tells you:** Where your documentation is missing. High values mean agents are making mistakes in areas where no reference material exists.

**What action it drives:** Extract the topics from uncovered findings and add them to your knowledge base. This is the single most direct way to improve agent quality.

**Formula:**

```
score = uncovered_findings / total_rework_findings
```

Where `uncovered_findings` = findings where no doc chunk matches at `strong` or `domain` tier (see [matching algorithm](#the-hybrid-matching-algorithm) below).

Lower is better.

**N/A conditions:** Returns `score=None` when:
- No sessions have rework findings (critical/major, non-synthesized), OR
- No sessions have knowledge context (docs not loaded, no `knowledge_context` in phases)

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.20 | KB covers most failure domains |
| Yellow | 0.20–0.40 | Notable gaps in knowledge coverage |
| Red | > 0.40 | KB is missing content for many failure modes |

**Pitfall:** Gap rate uses word overlap and path matching to determine coverage. Synonyms or domain-specific terminology in findings that does not appear in doc chunks will cause false negatives (findings appear uncovered when they are actually covered by differently-worded content). Review uncovered findings manually before investing in new documentation.

---

## knowledge_miss_rate — Knowledge miss rate

**What it measures:** Ratio of rework-triggering findings (critical/major) in domains that ARE covered by the knowledge base but the agent still got wrong.

**What it tells you:** How often the agent fails despite having the right reference material. High values may indicate the KB content is unclear, outdated, or the agent is not using it effectively.

**What action it drives:** Review the KB content for the affected domains. The information exists but is not preventing mistakes — it may need to be restructured, made more explicit, or moved closer to the agent's context window.

**Formula:**

```
score = covered_findings / total_rework_findings
```

Where `covered_findings` = findings where at least one doc chunk matches at `strong` or `domain` tier.

Lower is better.

**N/A conditions:** Returns `score=None` when:
- No sessions have rework findings (critical/major, non-synthesized), OR
- No sessions have knowledge context (docs not loaded, no `knowledge_context` in phases)

| Zone | Range | Meaning |
|------|-------|---------|
| Green | < 0.10 | Agent uses KB content effectively |
| Yellow | 0.10–0.30 | Agent sometimes ignores available knowledge |
| Red | > 0.30 | Agent frequently fails despite having KB coverage |

---

## Relationship between gap rate and miss rate

These two metrics are complementary. When both metrics produce scores from the same run, `knowledge_gap_rate + knowledge_miss_rate = 1.0` exactly — each finding is classified as either covered or uncovered, never both or neither. Scores that appear to not sum to 1.0 come from comparing across different runs with different doc paths or session sets.

Reading the two together:

- **High gap rate, low miss rate:** Your KB works well where it exists — expand its coverage.
- **Low gap rate, high miss rate:** Your KB covers the right domains but the content is not effective — improve quality.
- **Both high:** Both coverage and quality need work.

---

## Worked example

Consider a session with two critical findings evaluated against a docs directory:

```
docs/
  auth/
    permissions.md    # "## Delete permissions\nAlways check delete endpoint access..."
  database/
    schema.md         # "## Schema migrations\nAlways run makemigrations before migrate..."
```

**Finding 1:** `{file: "src/auth/views.py", issue: "Missing permission check on delete endpoint", severity: "critical"}`

1. Path match vs `auth/permissions.md`: `src/auth/views.py` shares `auth` with `auth/permissions.md` → **True**
2. Word match: finding tokens = `{missing, permission, check, delete, endpoint}`, chunk tokens = `{delete, permissions, check, endpoint, access}`, shared = `{delete, check, endpoint}` (3 words) → **True**
3. Tier: **strong** (both signals)
4. Coverage: **covered** → counts toward `knowledge_miss_rate`

**Finding 2:** `{file: "src/billing/invoice.py", issue: "Tax calculation uses wrong rounding mode", severity: "critical"}`

1. Path match vs `auth/permissions.md`: no shared components → **False**
2. Path match vs `database/schema.md`: no shared components → **False**
3. Word match vs both chunks: `{tax, calculation, uses, wrong, rounding, mode}` shares < 3 words with either → **False**
4. Tier: **none** for all chunks
5. Coverage: **uncovered** → counts toward `knowledge_gap_rate`

**Result:** `knowledge_gap_rate = 1/2 = 0.50`, `knowledge_miss_rate = 1/2 = 0.50`. Sum = 1.0.

---

## How documentation is loaded

RAKI walks the `--docs-path` directory recursively and loads files matching the configured extensions (default: `.md`, `.txt`). Each file goes through safety checks before loading:

- **Symlink rejection**: symlinks (both files and ancestor directories) are skipped to prevent path escape
- **Path traversal guard**: the resolved path must be under the docs root
- **Per-file size limit**: files over 1 MB are skipped (override via the `max_file_size` parameter in `load_docs()` — not yet exposed as a CLI flag)
- **Total size limit**: loading stops at 50 MB total (override via the `max_total_size` parameter in `load_docs()` — not yet exposed as a CLI flag)

### Chunking

Each file is split into chunks based on its format:

- **Markdown** (`.md`): split on heading patterns (`# `, `## `, `### `, etc.). Each heading starts a new chunk that includes the heading text. Content before the first heading becomes a preamble chunk.
- **RST** (`.rst`): split on RST underline headings (`=`, `-`, `~`, etc.). A heading is detected as a text line followed by a line of heading characters at least as long as the text.
- **Plain text** (`.txt`): split on double newlines (paragraph boundaries). Small paragraphs are merged up to 2000 characters per chunk.

**Note on chunk granularity:** The chunking strategy affects word match precision. A very long markdown section (no sub-headings) becomes one large chunk with many tokens, increasing the chance of word overlap with unrelated findings. A heavily-subdivided doc with many headings produces smaller, focused chunks with fewer false-positive word matches. This means knowledge metric scores can be sensitive to documentation *formatting*, not just content — adding headings to a doc may change scores even though the content is unchanged.

### Domain extraction

Each chunk is tagged with a **domain** derived from the directory structure. The first subdirectory under the docs root becomes the domain:

```
docs/auth/setup.md       → domain "auth"
docs/api/v2/endpoints.md → domain "api"
docs/readme.md           → domain "general"
```

Files directly in the docs root get the domain `"general"`.

The domain tag is organizational metadata on each `DocChunk` — it provides a way to group and report on documentation by topic area. It is **not used by the matching algorithm** directly. The matching algorithm operates on `chunk.source_file` (full relative path) for path matching and `chunk.text` for word matching — it does not reference the `domain` field.

---

## The hybrid matching algorithm

When computing knowledge metrics, RAKI determines whether each review finding is **covered** by the knowledge base. Matching uses a two-signal system that combines **path matching** and **word matching** to produce a confidence tier for each (finding, doc chunk) pair.

### Signal 1: Path matching

```python
path_match(finding.file, chunk.source_file) → bool
```

Compares the path components of the finding's file location against the path components of the doc chunk's source file. Returns `True` when they share at least one path component. Components are split using `pathlib.Path.parts` — each directory name and the filename (with extension) is a discrete component.

**Examples:**

| Finding file | Chunk source | Match? | Shared component |
|---|---|---|---|
| `src/auth/views.py` | `auth/setup.md` | Yes | `auth` |
| `src/auth/views.py` | `database/schema.md` | No | — |
| `src/api/endpoints.py` | `api/v2/reference.md` | Yes | `api` |
| `None` | `auth/setup.md` | No | — (finding has no file) |
| `src/auth/views.py` | `src/troubleshooting.md` | Yes | `src` ⚠ false positive |

Path matching is the **high-confidence signal** because it links findings to documentation about the same code area. A finding in `src/auth/views.py` is likely relevant to docs in the `auth/` directory regardless of word overlap.

**Caution:** Common structural prefixes like `src`, `lib`, `internal`, `pkg`, `cmd` can cause false-positive path matches (see [known limitations](#known-limitations)).

### Signal 2: Word matching

```python
word_match(finding.issue, chunk.text) → bool
```

Tokenizes both the finding's issue text and the chunk's text, removes stop words, and checks whether they share **at least 3 non-stop words**.

**Tokenization:**
- Extract alphabetic-only sequences via `re.findall(r"[a-z]+", text.lower())`
- This means numeric components are dropped: `v2` → `v`, `http2` → `http`, `3xx` → `xx`
- Remove ~180 English stop words (articles, prepositions, conjunctions, pronouns, auxiliaries, common adverbs)

The 3-word threshold prevents spurious matches from single shared words (which are common across unrelated domains) while still catching topical overlap when findings and docs discuss the same concepts.

**Examples:**

| Finding issue | Chunk text (excerpt) | Shared words | Match? |
|---|---|---|---|
| "Missing @pytest.mark.django_db on test_group_visibility" | "Django test decorators: always use django_db for database access" | `test`, `django`, `db` | Yes (3) |
| "SQL injection in user query" | "Authentication flow: login, logout, session..." | `user` | No (1) |

### Confidence tiers

The two signals combine into four tiers:

| Tier | Path match | Word match (≥3) | Meaning |
|---|---|---|---|
| **strong** | ✓ | ✓ | Both signals agree — high confidence the KB covers this domain |
| **domain** | ✓ | ✗ | Same code area but different topic — KB has domain coverage |
| **content** | ✗ | ✓ | Topical word overlap but different code area — weak signal |
| **none** | ✗ | ✗ | No relationship found |

**Only `strong` and `domain` tiers count as "covered."**

The `content` tier (word overlap only) is explicitly excluded because it produced too many false positives in practice — common programming terms (`test`, `error`, `function`, `class`) appear across unrelated doc chunks, causing nearly every finding to match. The false-positive rate on the `content` tier was observed empirically during development but was not formally measured with a holdout set. Requiring a path match anchors the coverage signal to a specific code area.

### Coverage decision

A finding is considered **covered** by the KB when at least one doc chunk matches at the `strong` or `domain` tier:

```python
is_finding_covered_by_chunks(finding, doc_chunks) → bool
# Returns True if any chunk matches at "strong" or "domain" tier
```

### Finding filtering

Both knowledge metrics only consider:
- **Critical and major findings** — minor findings are excluded because they rarely indicate knowledge gaps
- **Non-synthesized findings** — synthesized findings (inferred from tool failures) match too broadly against doc chunks and would inflate rates artificially
- **Sessions with rework** — only sessions where `rework_cycles > 0` are examined (clean sessions have no findings to classify)

**Implementation note:** On the doc-chunk path, synthesized findings are filtered per-finding (`finding.finding_source == "synthesized"`). On the legacy path, synthesized contexts are filtered per-session (`sample.context_source == "synthesized"`), which skips the entire session's findings rather than individual ones. The net effect is similar but the granularity differs.

---

## Legacy matching path

When no doc chunks are available (no `--docs-path` provided), RAKI falls back to a simpler matching algorithm using the `knowledge_context` string embedded in session phases:

1. Extract `knowledge_context` from the latest `implement` or `session` phase
2. Tokenize both the finding's issue text and the knowledge context (same stop-word removal)
3. A finding is "covered" if the token sets share **any** non-stop words (minimum overlap ≥ 1, unlike the 3-word threshold on the doc-chunk path)

This legacy path exists for backward compatibility with sessions that carry inline knowledge context. It is less precise than the doc-chunk path because it compares against a single merged text blob rather than per-domain chunks, and the lower overlap threshold produces more false positives.

Sessions with `context_source == "synthesized"` are skipped entirely on the legacy path to avoid inflated matching from synthetic contexts.

---

## Debugging knowledge metrics

When knowledge metrics return unexpected values:

**`knowledge_gap_rate = 1.0` (everything uncovered) despite having docs:**

1. Verify docs are loading: check for `Skipping` warnings in verbose output (`-v` flag)
2. Check directory structure: are docs all in the root? (domain = `"general"` for everything, see [flat docs limitation](#known-limitations))
3. Check that doc file paths share components with finding file paths — if your source is under `src/auth/` but docs are under `docs/authentication/`, path matching won't connect them
4. Check that findings have `file` fields populated — findings without a file path cannot match via path, only via 3-word overlap

**`knowledge_miss_rate = 0.0` (everything uncovered) but you expected coverage:**

- This is the same as `knowledge_gap_rate = 1.0`. See above.

**Metrics return N/A:**

- No sessions have `rework_cycles > 0`, OR
- No sessions have knowledge context (check `--docs-path` was provided), OR
- All findings are minor or synthesized (only critical/major non-synthesized findings are evaluated)

---

## Matching algorithm in detail

For contributors working on the matching code, here is the complete algorithm as implemented in `src/raki/metrics/knowledge/_common.py`:

### `tokenize(text) → set[str]`

```
1. Lowercase the input text
2. Extract all sequences of alphabetic characters: re.findall(r"[a-z]+", text)
3. Remove tokens that appear in STOP_WORDS (~180 common English words)
4. Return the remaining tokens as a set
```

Note: numeric characters are not alphabetic, so tokens like `v2`, `http2`, `3xx` lose their numeric components (`v`, `http`, `xx`). Version identifiers and numeric codes are not preserved.

### `path_match(finding_file, chunk_source) → bool`

```
1. If finding_file is None → return False
2. Split finding_file into path components (via pathlib.Path.parts)
3. Split chunk_source into path components
4. Return True if the two sets of components share any element
```

Components include the filename with extension: `Path("src/auth/views.py").parts` = `("src", "auth", "views.py")`.

### `word_match(finding_text, chunk_text) → bool`

```
1. Tokenize finding_text → set A
2. Tokenize chunk_text → set B
3. Return True if |A ∩ B| >= 3
```

### `match_finding_to_chunk(finding, chunk) → tier`

```
1. has_path = path_match(finding.file, chunk.source_file)
2. has_word = word_match(finding.issue, chunk.text)
3. If has_path AND has_word → "strong"
4. If has_path → "domain"
5. If has_word → "content"
6. Otherwise → "none"
```

### `is_finding_covered_by_chunks(finding, doc_chunks) → bool`

```
1. For each chunk in doc_chunks:
   a. tier = match_finding_to_chunk(finding, chunk)
   b. If tier is "strong" or "domain" → return True
2. Return False
```

### Design decisions

| Decision | Rationale |
|---|---|
| Require path match for coverage | Word-only matching (`content` tier) produced too many false positives in practice — not formally measured but observed empirically during development |
| 3-word minimum for word match | 1-2 word overlap catches too many coincidental matches between unrelated domains |
| Stop-word list (~180 words) | Comprehensive enough to filter noise, focused on English function words that carry no domain signal |
| Path component matching (not substring) | Prevents `auth` in `docs/auth/` from matching `authorization.py` — path components are discrete tokens |
| Only critical/major findings | Minor findings (style nits, naming) rarely indicate knowledge gaps worth documenting |
| Exclude synthesized findings | Synthesized findings from tool output contain raw error text that matches broadly and inflates rates |
| Domain field not used in matching | Domain extraction is for organizational metadata; matching uses full path components for finer granularity |

### Known limitations

1. **Common path prefixes:** Projects where both source code and docs share structural prefixes (`src/`, `lib/`, `internal/`, `pkg/`, `cmd/`) will get false-positive path matches on those components. For example, `src/auth/views.py` matches `src/troubleshooting.md` on the `src` component even though they're in unrelated domains. Mitigation: organize docs without mirroring source tree prefixes, or place docs in a separate root (e.g., `docs/auth/` not `src/docs/auth/`).
2. **Synonyms:** "authentication" in findings won't match "auth" in doc paths. Path matching helps here (both would share the `auth` component if the directory is named `auth`), but word matching misses synonyms entirely.
3. **Short findings:** Findings with fewer than 3 meaningful words after stop-word removal cannot match via word overlap. They can still match via path if they have a `file` field.
4. **Flat docs structure:** If all docs are in the root directory (domain = `"general"` for everything), path matching loses its discriminating power. Every finding's file path components are compared against root-level filenames, which rarely share meaningful components. Organize docs into subdirectories matching your code's domain structure.
5. **Non-English content:** The stop-word list is English-only. Non-English findings and docs will retain more tokens, potentially increasing false-positive word matches.
6. **Numeric tokens dropped:** The tokenizer only extracts alphabetic characters. Findings about "HTTP/2 support" or "Python 3.14 compatibility" lose their version identifiers, reducing matching precision for version-specific content.
7. **Chunk size sensitivity:** Large markdown sections without sub-headings become single large chunks with many tokens, increasing false-positive word matches. Adding headings to docs can change metric scores without changing content. Prefer well-structured docs with clear heading hierarchy.
