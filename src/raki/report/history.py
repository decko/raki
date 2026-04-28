"""JSONL history log — append one compact record per evaluation run for cross-run tracking."""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from raki.model.report import EvalReport


def _git_sha() -> str | None:
    """Return the short git SHA of HEAD, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _config_hash(config: dict) -> str:
    """Return a SHA-256 hex digest of the config dict, sorted for determinism."""
    serialized = json.dumps(sorted(config.items()))
    return hashlib.sha256(serialized.encode()).hexdigest()


class HistoryEntry(BaseModel):
    """Compact record written to the JSONL history file after each evaluation run.

    Each line in ``history.jsonl`` is a JSON-serialised ``HistoryEntry``.
    The entry contains only the aggregate view of a run — no raw session data —
    so the file stays small and readable over many runs.
    """

    schema_version: int = 1
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sessions_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    manifest: str | None = None
    config_hash: str = ""
    git_sha: str | None = None
    warning_count: int = 0


def append_history_entry(
    report: EvalReport,
    history_path: Path,
    *,
    sessions_count: int,
    manifest_file: Path | None = None,
) -> None:
    """Append a single ``HistoryEntry`` line to the JSONL history file.

    Args:
        report: The completed evaluation report for this run.
        history_path: Path to the JSONL history file.  Created (with parent
            directories) on first call.  Subsequent calls append without
            overwriting existing entries.
        sessions_count: Number of sessions that were evaluated in this run.
        manifest_file: Path to the manifest file used for this run (basename stored).

    Raises:
        ValueError: If ``history_path`` is a symlink (security guard).
    """
    if history_path.is_symlink():
        raise ValueError(f"Refusing to write to symlink: {history_path}")
    history_path = history_path.resolve()

    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metrics dict excluding None values
    metrics_dict = {
        key: value for key, value in report.aggregate_scores.items() if value is not None
    }

    entry = HistoryEntry(
        run_id=report.run_id,
        timestamp=report.timestamp,
        sessions_count=sessions_count,
        metrics=metrics_dict,
        manifest=manifest_file.name if manifest_file is not None else None,
        config_hash=_config_hash(report.config),
        git_sha=_git_sha(),
        warning_count=len(report.warnings),
    )

    line = json.dumps(entry.model_dump(mode="json"), default=str)
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(line + "\n")


def load_run_ids(history_path: Path) -> set[str]:
    """Return the set of ``run_id`` values already present in *history_path*.

    This is a lightweight deduplication helper for ``import-history``: callers
    can check membership in O(1) before attempting to write a new entry.

    Args:
        history_path: Path to the JSONL history file.  A missing file is
            treated as an empty history — an empty set is returned.

    Returns:
        Set of run_id strings found in the existing history.

    Raises:
        ValueError: If ``history_path`` is a symlink (security guard).
    """
    return {entry.run_id for entry in load_history(history_path)}


def import_history_entry(
    entry: HistoryEntry,
    history_path: Path,
    existing_ids: set[str],
) -> bool:
    """Append *entry* to the JSONL history file if its ``run_id`` is not already present.

    The caller owns the *existing_ids* set and is responsible for keeping it
    up-to-date: this function **adds** ``entry.run_id`` to *existing_ids* when
    a write occurs, so that repeated calls within the same import session remain
    idempotent without re-reading the file each time.

    Args:
        entry: The :class:`HistoryEntry` to append.
        history_path: Path to the JSONL history file.  Parent directories are
            created on first write.
        existing_ids: Mutable set of run_ids already in the file.  Updated
            in-place when an entry is written.

    Returns:
        ``True`` if the entry was written, ``False`` if it was a duplicate.

    Raises:
        ValueError: If ``history_path`` is a symlink (security guard).
    """
    if entry.run_id in existing_ids:
        return False

    if history_path.is_symlink():
        raise ValueError(f"Refusing to write to symlink: {history_path}")

    resolved = history_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(entry.model_dump(mode="json"), default=str)
    with resolved.open("a", encoding="utf-8") as history_file:
        history_file.write(line + "\n")

    existing_ids.add(entry.run_id)
    return True


def load_history(history_path: Path) -> list[HistoryEntry]:
    """Load all history entries from a JSONL file.

    Malformed lines (invalid JSON or validation errors) are skipped with a
    warning rather than raising, so a single corrupt line does not prevent
    loading the rest of the history.

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
    for line_number, raw_line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            entries.append(HistoryEntry.model_validate(parsed))
        except (json.JSONDecodeError, ValidationError) as exc:
            warnings.warn(
                f"Skipping malformed history line {line_number} in {history_path}: {exc}",
                stacklevel=2,
            )
    return entries
