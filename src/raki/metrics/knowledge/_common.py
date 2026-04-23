"""Shared helpers for knowledge tier metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from raki.model import EvalSample

if TYPE_CHECKING:
    from raki.docs.chunker import DocChunk
    from raki.model import ReviewFinding

# Minimum word length to consider for overlap matching.
# Short words like "the", "and", "with" produce false-positive overlaps.
_MIN_WORD_LENGTH = 5

# Common English stop words that carry no domain signal.
STOP_WORDS: frozenset[str] = frozenset(
    {
        # Articles and determiners
        "a",
        "an",
        "the",
        # Prepositions
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "as",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "between",
        "beyond",
        "by",
        "down",
        "during",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "since",
        "through",
        "throughout",
        "to",
        "toward",
        "under",
        "until",
        "up",
        "upon",
        "with",
        "within",
        "without",
        # Conjunctions
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "whether",
        "not",
        "if",
        "than",
        "that",
        "though",
        "although",
        "because",
        "unless",
        "while",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        # Pronouns
        "all",
        "any",
        "each",
        "few",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "none",
        "our",
        "ours",
        "she",
        "some",
        "such",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "us",
        "very",
        "we",
        "what",
        "you",
        "your",
        # Auxiliaries and common verbs
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "get",
        "got",
        "had",
        "has",
        "have",
        "having",
        "is",
        "may",
        "might",
        "must",
        "need",
        "ought",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
        # Common adverbs / other function words
        "also",
        "always",
        "back",
        "even",
        "ever",
        "however",
        "just",
        "never",
        "now",
        "once",
        "only",
        "other",
        "own",
        "same",
        "then",
        "too",
        "well",
        # One/two-letter words not caught above
        "no",
        "ok",
        "re",
    }
)


def tokenize(text: str) -> set[str]:
    """Return non-stop-word tokens from *text* as a lower-cased set.

    Splits on non-alphabetic characters and removes tokens that appear in
    STOP_WORDS.  Used by both the doc-chunk path and the legacy
    knowledge-context path.
    """
    return {word for word in re.findall(r"[a-z]+", text.lower()) if word not in STOP_WORDS}


def word_match(finding_text: str, chunk_text: str) -> bool:
    """Return True when *finding_text* and *chunk_text* share ≥ 3 non-stop words.

    Overlap is computed on lower-cased alphabetic tokens after removing STOP_WORDS.
    """
    return len(tokenize(finding_text) & tokenize(chunk_text)) >= 3


def path_match(finding_file: str | None, chunk_source: str) -> bool:
    """Return True when *finding_file* shares at least one path component with *chunk_source*.

    Path components are derived via :class:`pathlib.Path`.  Returns False
    immediately when *finding_file* is None so callers can pass
    ``finding.file`` directly without a prior None-check.

    Examples::

        path_match("src/auth/views.py", "auth/setup.md")   # True  — "auth"
        path_match("src/auth/views.py", "database/schema.md")  # False
        path_match(None, "auth/setup.md")                  # False
    """
    if finding_file is None:
        return False
    finding_parts = set(Path(finding_file).parts)
    chunk_parts = set(Path(chunk_source).parts)
    return bool(finding_parts & chunk_parts)


def match_finding_to_chunk(
    finding: ReviewFinding,
    chunk: DocChunk,
) -> Literal["strong", "domain", "content", "none"]:
    """Return the coverage tier for a (finding, chunk) pair.

    Tiers (highest to lowest):

    * ``"strong"``  — path match **and** word overlap ≥ 3
    * ``"domain"``  — path match only
    * ``"content"`` — word overlap ≥ 3 only (path match absent)
    * ``"none"``    — neither criterion satisfied

    Only ``"strong"`` and ``"domain"`` are treated as *covered* by
    :func:`is_finding_covered_by_chunks`.  ``"content"`` was the old
    (too-loose) behaviour that this system replaces.
    """
    has_path = path_match(finding.file, chunk.source_file)
    has_word = word_match(finding.issue, chunk.text)
    if has_path and has_word:
        return "strong"
    if has_path:
        return "domain"
    if has_word:
        return "content"
    return "none"


def extract_knowledge_context(sample: EvalSample) -> str | None:
    """Extract knowledge_context text from the latest implement or session phase.

    Returns the lowercased knowledge_context string, or None when no
    matching phase carries knowledge_context.
    """
    matching = [phase for phase in sample.phases if phase.name in ("implement", "session")]
    if not matching:
        return None
    phase = max(matching, key=lambda phase: phase.generation)
    if phase.knowledge_context is None:
        return None
    return phase.knowledge_context.lower()


def build_domain_word_sets(doc_chunks: list[DocChunk]) -> dict[str, set[str]]:
    """Build per-domain word sets from doc chunks.

    Groups chunks by domain, then collects all words (>= _MIN_WORD_LENGTH chars)
    from each domain's chunk texts. Returns a mapping of domain -> set of words.

    Used by knowledge metrics for domain-aware overlap matching.
    """
    domain_words: dict[str, set[str]] = {}
    for chunk in doc_chunks:
        words = {word for word in chunk.text.lower().split() if len(word) >= _MIN_WORD_LENGTH}
        if chunk.domain not in domain_words:
            domain_words[chunk.domain] = words
        else:
            domain_words[chunk.domain] |= words
    return domain_words


def is_finding_covered_by_chunks(issue_text: str, domain_word_sets: dict[str, set[str]]) -> bool:
    """Check whether a finding's issue text overlaps with any domain's word set.

    A finding is "covered" when its significant words (>= _MIN_WORD_LENGTH chars)
    have a non-empty intersection with at least one domain's word set.

    This produces domain-aware matching: a finding about "authentication" will only
    match if some domain's docs contain authentication-related words, rather than
    matching against a merged blob of all docs.
    """
    issue_words = {word for word in issue_text.lower().split() if len(word) >= _MIN_WORD_LENGTH}
    return any(issue_words & words for words in domain_word_sets.values())
