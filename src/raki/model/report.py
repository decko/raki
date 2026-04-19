from datetime import datetime, timezone

from pydantic import BaseModel, Field

from raki.model.dataset import EvalSample


class MetricResult(BaseModel):
    name: str
    score: float
    details: dict = Field(default_factory=dict)
    sample_scores: dict[str, float] = Field(default_factory=dict)


class SampleResult(BaseModel):
    sample: EvalSample
    scores: list[MetricResult]


class EvalReport(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: dict = Field(default_factory=dict)
    aggregate_scores: dict[str, float] = Field(default_factory=dict)
    sample_results: list[SampleResult] = Field(default_factory=list)
    manifest_hash: str | None = None
