from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SessionEvent(BaseModel):
    timestamp: datetime
    phase: str | None = None
    kind: Literal[
        "phase_started",
        "phase_completed",
        "phase_failed",
        "review_merged",
        "review_rework_routed",
        "rework_feedback_injected",
        "reviewer_started",
        "reviewer_completed",
    ]
    data: dict = Field(default_factory=dict)
