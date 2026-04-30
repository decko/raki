from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from raki.model.events import SessionEvent
from raki.model.ground_truth import GroundTruth
from raki.model.phases import PhaseResult, ReviewFinding


class SessionMeta(BaseModel):
    session_id: str
    tenant_id: str | None = None
    ticket: str | None = None
    started_at: datetime
    total_cost_usd: float | None = None
    total_phases: int
    rework_cycles: int
    knowledge_version: str | None = None
    model_id: str | None = None
    # Pipeline/orchestrator metadata (ticket #175)
    orchestrator: str | None = None  # e.g. "soda", "bridge", "alcove"
    provider: str | None = (
        None  # LLM provider reported by the orchestrator (distinct from MetricConfig.llm_provider)
    )
    pipeline_phases: list[str] | None = None  # ordered phase names from the session
    adapter_format: str = ""  # adapter name used to load this session (ticket #236)


class EvalSample(BaseModel):
    session: SessionMeta
    phases: list[PhaseResult]
    findings: list[ReviewFinding]
    events: list[SessionEvent]
    ground_truth: GroundTruth | None = None
    context_source: Literal["explicit", "synthesized"] | None = None


class EvalDataset(BaseModel):
    samples: list[EvalSample]
    manifest_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
