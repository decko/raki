from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from raki.model import EvalDataset
from raki.model.report import MetricResult

LLMProvider = Literal["vertex-anthropic", "anthropic", "google"]


@dataclass
class TokenAccumulator:
    """Accumulates token usage across LLM judge calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class MetricConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    llm_provider: LLMProvider = "vertex-anthropic"
    llm_model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int | None = None
    batch_size: int = 4
    judge_log_path: Path | None = None  # path to judge_log.jsonl for audit logging
    project_root: Path | None = None  # root directory for path validation
    doc_chunks: list[Any] = Field(default_factory=list)  # list[DocChunk], Any to avoid import
    token_accumulator: TokenAccumulator | None = Field(default=None, repr=False)


@runtime_checkable
class Metric(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def requires_ground_truth(self) -> bool: ...

    @property
    def requires_llm(self) -> bool: ...

    @property
    def higher_is_better(self) -> bool: ...

    @property
    def display_format(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def rationale(self) -> str: ...

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult: ...
