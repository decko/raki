"""Shared helpers for knowledge tier metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raki.model import EvalSample

if TYPE_CHECKING:
    from raki.docs.chunker import DocChunk

# Minimum word length to consider for overlap matching.
# Short words like "the", "and", "with" produce false-positive overlaps.
_MIN_WORD_LENGTH = 5


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
