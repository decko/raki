from raki.metrics.knowledge.gap_rate import KnowledgeGapRate
from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

ALL_KNOWLEDGE: tuple[KnowledgeGapRate, KnowledgeMissRate] = (
    KnowledgeGapRate(),
    KnowledgeMissRate(),
)

__all__ = [
    "ALL_KNOWLEDGE",
    "KnowledgeGapRate",
    "KnowledgeMissRate",
]
