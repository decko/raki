from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from raki.model import EvalDataset
from raki.model.report import MetricResult


class MetricConfig(BaseModel):
    llm_provider: str = "vertex-anthropic"
    llm_model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    batch_size: int = 4
    judge_log_path: Path | None = None  # path to judge_log.jsonl for audit logging


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

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult: ...
