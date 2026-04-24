"""JSONL history log — append one compact record per evaluation run for cross-run tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from raki.model.report import EvalReport


class HistoryEntry(BaseModel):
    """Compact record written to the JSONL history file after each evaluation run.

    Each line in ``raki-history.jsonl`` is a JSON-serialised ``HistoryEntry``.
    The entry contains only the aggregate view of a run — no raw session data —
    so the file stays small and readable over many runs.
    """

    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_count: int
    aggregate_scores: dict[str, float | None] = Field(default_factory=dict)
    manifest_hash: str | None = None


def append_history_entry(
    report: EvalReport,
    history_path: Path,
    *,
    session_count: int,
) -> None:
    """Append a single ``HistoryEntry`` line to the JSONL history file.

    Args:
        report: The completed evaluation report for this run.
        history_path: Path to the JSONL history file.  Created (with parent
            directories) on first call.  Subsequent calls append without
            overwriting existing entries.
        session_count: Number of sessions that were evaluated in this run.

    Raises:
        ValueError: If ``history_path`` is a symlink (security guard).
    """
    if history_path.is_symlink():
        raise ValueError(f"Refusing to write to symlink: {history_path}")
    history_path = history_path.resolve()

    history_path.parent.mkdir(parents=True, exist_ok=True)

    entry = HistoryEntry(
        run_id=report.run_id,
        timestamp=report.timestamp,
        session_count=session_count,
        aggregate_scores=dict(report.aggregate_scores),
        manifest_hash=report.manifest_hash,
    )

    line = json.dumps(entry.model_dump(mode="json"), default=str)
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(line + "\n")


def load_history(history_path: Path) -> list[HistoryEntry]:
    """Load all history entries from a JSONL file.

    Args:
        history_path: Path to the JSONL history file.

    Returns:
        List of ``HistoryEntry`` objects in file order (oldest first).
        Returns an empty list when the file does not exist.

    Raises:
        ValueError: If ``history_path`` is a symlink (security guard).
    """
    if history_path.is_symlink():
        raise ValueError(f"Refusing to read symlink: {history_path}")

    if not history_path.exists():
        return []

    entries: list[HistoryEntry] = []
    for raw_line in history_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        entries.append(HistoryEntry.model_validate(json.loads(stripped)))
    return entries
