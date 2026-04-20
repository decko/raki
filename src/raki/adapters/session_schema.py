import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from raki.adapters.redact import redact_dict, redact_sensitive
from raki.model import EvalSample, PhaseResult, ReviewFinding, SessionEvent, SessionMeta

PHASE_NAMES = ["triage", "plan", "implement", "verify"]

MAX_SESSION_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _read_bounded(path: Path) -> str:
    """Read file contents, raising ValueError if the file exceeds MAX_SESSION_FILE_SIZE."""
    file_size = path.stat().st_size
    if file_size > MAX_SESSION_FILE_SIZE:
        raise ValueError(
            f"File {path} is {file_size} bytes, exceeding the {MAX_SESSION_FILE_SIZE} byte limit"
        )
    return path.read_text(encoding="utf-8")


def _validate_path(file_path: Path, session_dir: Path) -> None:
    """Reject paths that escape the session directory (e.g. via symlinks)."""
    resolved = file_path.resolve()
    if not resolved.is_relative_to(session_dir.resolve()):
        raise ValueError(
            f"Path {file_path} resolves to {resolved}, which is outside "
            f"session directory {session_dir.resolve()}"
        )


def _extract_model_id_from_events(events: list[SessionEvent]) -> str | None:
    """Extract model_id from the first event that has a 'model' key in its data."""
    for event in events:
        model = event.data.get("model")
        if model and isinstance(model, str):
            return model
    return None


def _extract_token_counts_from_events(
    events: list[SessionEvent], phase_name: str
) -> dict[str, int | None]:
    """Extract tokens_in/tokens_out from phase_completed events for a given phase.

    Returns a dict with 'tokens_in' and 'tokens_out' keys, values may be None.
    """
    for event in events:
        if event.phase == phase_name and event.kind == "phase_completed":
            tokens_in = event.data.get("tokens_in")
            tokens_out = event.data.get("tokens_out")
            if tokens_in is not None or tokens_out is not None:
                return {"tokens_in": tokens_in, "tokens_out": tokens_out}
    return {"tokens_in": None, "tokens_out": None}


class SessionSchemaAdapter:
    name: str = "session-schema"
    description: str = "Directory-based session format with structured phases"
    detection_hint: str = "meta.json + events.jsonl"

    def detect(self, source: Path) -> bool:
        if source.is_symlink():
            return False
        return (source / "meta.json").exists() and (source / "events.jsonl").exists()

    def load(self, source: Path) -> EvalSample:
        if source.is_symlink():
            raise ValueError(f"Refusing to load symlink: {source}")
        meta_path = source / "meta.json"
        _validate_path(meta_path, source)
        meta_raw: dict[str, Any] = json.loads(_read_bounded(meta_path))
        session_id = str(meta_raw.get("ticket", source.name))
        phases_dict = meta_raw.get("phases") or {}

        events = self._load_events(source)

        # Extract model_id: prefer meta.json, fall back to events
        model_id = meta_raw.get("model_id")
        if not model_id:
            model_id = _extract_model_id_from_events(events)

        meta = SessionMeta(
            session_id=session_id,
            ticket=str(meta_raw.get("ticket")),
            started_at=meta_raw["started_at"],
            total_cost_usd=meta_raw.get("total_cost"),
            total_phases=len(phases_dict),
            rework_cycles=meta_raw.get("rework_cycles", 0),
            model_id=model_id,
        )
        phases = self._load_phases(source, meta_raw, events)
        findings = self._load_findings(source)
        return EvalSample(session=meta, phases=phases, findings=findings, events=events)

    def _load_phases(
        self,
        source: Path,
        meta_raw: dict[str, Any],
        events: list[SessionEvent],
    ) -> list[PhaseResult]:
        phases: list[PhaseResult] = []
        for phase_name in PHASE_NAMES:
            phases.extend(self._load_phase_files(source, phase_name, meta_raw, events))
        return phases

    def _load_phase_files(
        self,
        source: Path,
        phase_name: str,
        meta_raw: dict[str, Any],
        events: list[SessionEvent],
    ) -> list[PhaseResult]:
        results: list[PhaseResult] = []
        pattern = re.compile(rf"^{re.escape(phase_name)}\.json(\.\d+)?$")
        matched_files = sorted(
            [file_path for file_path in source.iterdir() if pattern.match(file_path.name)],
            key=lambda file_path: file_path.name,
        )

        # Collect suffixed generation numbers to compute base file generation
        suffixed_generations: list[int] = []
        for file_path in matched_files:
            suffix_match = re.search(r"\.(\d+)$", file_path.name)
            if suffix_match:
                suffixed_generations.append(int(suffix_match.group(1)))

        phases_dict = meta_raw.get("phases") or {}
        phase_meta = phases_dict.get(phase_name) or {}

        # Extract token counts from events as fallback
        event_tokens = _extract_token_counts_from_events(events, phase_name)

        # Determine tokens_in/tokens_out: prefer phase metadata, fall back to events
        meta_tokens_in = phase_meta.get("tokens_in")
        tokens_in = meta_tokens_in if meta_tokens_in is not None else event_tokens["tokens_in"]
        meta_tokens_out = phase_meta.get("tokens_out")
        tokens_out = meta_tokens_out if meta_tokens_out is not None else event_tokens["tokens_out"]

        for file_path in matched_files:
            _validate_path(file_path, source)
            suffix_match = re.search(r"\.(\d+)$", file_path.name)
            if suffix_match:
                generation = int(suffix_match.group(1))
            else:
                # Base file = latest generation
                # Use meta generation if available; otherwise max(suffixed) + 1 or 1
                meta_generation = phase_meta.get("generation")
                if meta_generation is not None:
                    generation = meta_generation
                elif suffixed_generations:
                    generation = max(suffixed_generations) + 1
                else:
                    generation = 1
            try:
                raw = json.loads(_read_bounded(file_path))
            except json.JSONDecodeError:
                continue
            output_text = redact_sensitive(json.dumps(raw))
            results.append(
                PhaseResult(
                    name=phase_name,
                    generation=generation,
                    status=phase_meta.get("status", "completed"),
                    cost_usd=phase_meta.get("cost"),
                    duration_ms=phase_meta.get("duration_ms"),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    output=output_text,
                    output_structured=redact_dict(raw),
                )
            )
        return results

    def _load_findings(self, source: Path) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        pattern = re.compile(r"^review\.json(\.\d+)?$")
        for file_path in sorted(source.iterdir()):
            if not pattern.match(file_path.name):
                continue
            _validate_path(file_path, source)
            try:
                raw = json.loads(_read_bounded(file_path))
            except json.JSONDecodeError:
                continue
            for finding_raw in raw.get("findings") or []:
                try:
                    findings.append(
                        ReviewFinding(
                            reviewer=finding_raw.get("source", "unknown"),
                            severity=finding_raw.get("severity", "minor"),
                            file=finding_raw.get("file"),
                            line=finding_raw.get("line"),
                            issue=redact_sensitive(finding_raw["issue"]),
                            suggestion=redact_sensitive(s)
                            if (s := finding_raw.get("suggestion")) is not None
                            else None,
                        )
                    )
                except (KeyError, ValueError):
                    continue
        return findings

    def _load_events(self, source: Path) -> list[SessionEvent]:
        events_file = source / "events.jsonl"
        if not events_file.exists():
            return []
        _validate_path(events_file, source)
        events: list[SessionEvent] = []
        for line in _read_bounded(events_file).strip().splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                events.append(
                    SessionEvent(
                        timestamp=raw["timestamp"],
                        phase=raw.get("phase") or None,
                        kind=raw["kind"],
                        data=redact_dict(raw.get("data", {})),
                    )
                )
            except (KeyError, ValueError, ValidationError):
                continue
        return events
