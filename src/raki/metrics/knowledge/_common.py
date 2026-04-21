"""Shared helpers for knowledge tier metrics."""

from raki.model import EvalSample


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
