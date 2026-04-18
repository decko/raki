from typing import Literal

from pydantic import BaseModel, Field


class GroundTruth(BaseModel):
    question: str | None = None
    expected_approach: str | None = None
    expected_files: list[str] | None = None
    expected_contexts: list[str] | None = None
    acceptance_criteria: list[str] | None = None
    reference_answer: str | None = None
    domains: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    knowledge_type: Literal["fact", "procedure", "constraint", "context-dependent"] | None = None
    expected_phase: str | None = None
