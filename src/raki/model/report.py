from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from raki.model.dataset import EvalSample


class MetricWarning(BaseModel):
    """A health-check warning produced for a single metric after evaluation.

    Warnings flag degenerate or dead metrics so operators can investigate data
    quality issues before drawing conclusions from scores.
    """

    metric_name: str
    check: str  # e.g. "dead_metric", "degenerate_metric"
    severity: Literal["warning", "error"]
    message: str


class MetricResult(BaseModel):
    name: str
    score: float | None = None
    details: dict = Field(default_factory=dict)
    sample_scores: dict[str, float] = Field(default_factory=dict)


class SampleResult(BaseModel):
    sample: EvalSample
    scores: list[MetricResult]


class EvalReport(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: dict = Field(default_factory=dict)
    aggregate_scores: dict[str, float | None] = Field(default_factory=dict)
    metric_details: dict[str, dict] = Field(default_factory=dict)
    sample_results: list[SampleResult] = Field(default_factory=list)
    manifest_hash: str | None = None
    warnings: list[MetricWarning] = Field(default_factory=list)
