from typing import Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict | None = None
    result_summary: str | None = None
    duration_ms: int | None = None
    token_cost: int | None = None


class PhaseResult(BaseModel):
    name: str
    generation: int
    status: Literal["completed", "failed", "skipped"]
    cost_usd: float | None = None
    duration_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    knowledge_context: str | None = None
    instruction_context: str | None = None
    output: str
    output_structured: dict | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    reviewer: str
    severity: Literal["critical", "major", "minor"]
    file: str | None = None
    line: int | None = None
    issue: str
    suggestion: str | None = None
    finding_source: Literal["review", "synthesized"] | None = None
