import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from raki.adapters.redact import redact_dict, redact_sensitive
from raki.model import EvalSample, PhaseResult, ReviewFinding, SessionEvent, SessionMeta

PHASE_NAMES = ["triage", "plan", "implement", "verify", "review", "submit", "monitor"]

MAX_SESSION_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_IMPLEMENT_FALLBACK_CHARS = 10_000
MAX_SYNTHESIZED_CONTEXT_CHARS = 50_000


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

    @staticmethod
    def _resolve_rework_cycles(meta_raw: dict[str, Any], phases_dict: dict[str, Any]) -> int:
        """Resolve rework_cycles from explicit meta or SODA phase generation numbers.

        If ``rework_cycles`` is present in *meta_raw* (even as 0) that value is
        used as-is for backward compatibility.  Otherwise the value is derived
        from the highest ``generation`` number found in *phases_dict*:
        ``max_generation - 1`` (generation 1 == first pass == 0 rework cycles).
        """
        if "rework_cycles" in meta_raw:
            return int(meta_raw["rework_cycles"])
        if not phases_dict:
            return 0
        max_generation = max(
            (phase.get("generation", 1) if isinstance(phase, dict) else 1)
            for phase in phases_dict.values()
        )
        return max(0, max_generation - 1)

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

        # Infer orchestrator from branch prefix, e.g. "soda/101" → "soda"
        branch: str | None = meta_raw.get("branch")
        orchestrator: str | None = branch.split("/")[0] if branch and "/" in branch else None
        pipeline_phases: list[str] | None = list(phases_dict.keys()) if phases_dict else None

        meta = SessionMeta(
            session_id=session_id,
            ticket=str(meta_raw.get("ticket")),
            started_at=meta_raw["started_at"],
            total_cost_usd=meta_raw.get("total_cost"),
            total_phases=len(phases_dict),
            rework_cycles=self._resolve_rework_cycles(meta_raw, phases_dict),
            model_id=model_id,
            orchestrator=orchestrator,
            pipeline_phases=pipeline_phases,
        )
        phases = self._load_phases(source, meta_raw, events)
        findings = self._load_findings(source)
        sample = EvalSample(session=meta, phases=phases, findings=findings, events=events)

        # Synthesize context from phase outputs if no phase has explicit knowledge_context
        has_explicit_context = any(phase.knowledge_context is not None for phase in sample.phases)
        if not has_explicit_context and sample.phases:
            synthesized = self._synthesize_context(sample.phases)
            if synthesized:
                # Store on the implement phase (where to_ragas_rows reads from)
                target_phase = None
                for phase in reversed(sample.phases):
                    if phase.name == "implement":
                        target_phase = phase
                        break
                if target_phase is None:
                    for phase in reversed(sample.phases):
                        if phase.name == "session":
                            target_phase = phase
                            break
                if target_phase is None and sample.phases:
                    target_phase = sample.phases[-1]
                if target_phase:
                    target_phase.knowledge_context = synthesized
                    sample.context_source = "synthesized"

        return sample

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

    # Map SODA review schema severity labels to raki ReviewFinding literals.
    # SODA uses uppercase (CRITICAL/IMPORTANT/MINOR); raki uses lowercase.
    # SODA "IMPORTANT" maps to raki "major" (there is no direct equivalent).
    _SODA_SEVERITY_MAP: dict[str, str] = {
        "CRITICAL": "critical",
        "IMPORTANT": "major",
        "MINOR": "minor",
        # Passthrough for pre-existing lowercase values (flat-findings format).
        "critical": "critical",
        "major": "major",
        "minor": "minor",
    }

    def _normalize_severity(self, raw_severity: str | None) -> str:
        """Normalize a raw severity string to a ReviewFinding-compatible literal.

        Returns 'minor' when the value is absent or unrecognised so that
        malformed review files never cause a ValidationError.
        """
        if raw_severity is None:
            return "minor"
        return self._SODA_SEVERITY_MAP.get(str(raw_severity), "minor")

    def _findings_from_flat(self, raw: dict) -> list[ReviewFinding]:
        """Extract findings from the legacy flat ``findings`` array format."""
        results: list[ReviewFinding] = []
        for finding_raw in raw.get("findings") or []:
            try:
                results.append(
                    ReviewFinding(
                        reviewer=finding_raw.get("source", "unknown"),
                        severity=self._normalize_severity(finding_raw.get("severity")),
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
        return results

    def _findings_from_perspectives(self, raw: dict) -> list[ReviewFinding]:
        """Extract findings from the SODA ``perspectives`` array format.

        Each perspective element has a ``name`` (e.g. ``"python"``) and a nested
        ``findings`` array.  The perspective name is used as the reviewer
        identifier, and SODA severity labels (CRITICAL/IMPORTANT/MINOR) are
        normalised to raki's lowercase literals (critical/major/minor).
        """
        results: list[ReviewFinding] = []
        for perspective in raw.get("perspectives") or []:
            if not isinstance(perspective, dict):
                continue
            reviewer = perspective.get("name", "unknown")
            for finding_raw in perspective.get("findings") or []:
                if not isinstance(finding_raw, dict):
                    continue
                try:
                    results.append(
                        ReviewFinding(
                            reviewer=str(reviewer),
                            severity=self._normalize_severity(finding_raw.get("severity")),
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
            if raw.get("perspectives") is not None:
                # SODA format: findings are nested inside perspective objects.
                findings.extend(self._findings_from_perspectives(raw))
            else:
                # Legacy flat format: top-level "findings" array.
                findings.extend(self._findings_from_flat(raw))
        return findings

    def _synthesize_context(self, phases: list[PhaseResult]) -> str | None:
        """Synthesize retrieval context from structured phase outputs.

        Extracts relevant fields from triage, plan, and implement phases:
        - Triage: approach, code_area, files, risks
        - Plan: approach, task descriptions and files
        - Implement: files_changed, commits, deviations; falls back to output text
        """
        context_chunks: list[str] = []

        for phase in phases:
            if phase.name == "triage" and phase.output_structured:
                triage_parts: list[str] = []
                approach = phase.output_structured.get("approach")
                if approach and isinstance(approach, str):
                    triage_parts.append(f"Approach: {approach}")
                code_area = phase.output_structured.get("code_area")
                if code_area and isinstance(code_area, str):
                    triage_parts.append(f"Code area: {code_area}")
                files = phase.output_structured.get("files")
                if files and isinstance(files, list):
                    triage_parts.append(f"Files: {', '.join(str(filepath) for filepath in files)}")
                risks = phase.output_structured.get("risks")
                if risks and isinstance(risks, list):
                    triage_parts.append(f"Risks: {', '.join(str(risk) for risk in risks)}")
                if triage_parts:
                    context_chunks.append(redact_sensitive("\n".join(triage_parts)))

            elif phase.name == "plan" and phase.output_structured:
                plan_parts: list[str] = []
                approach = phase.output_structured.get("approach")
                if approach and isinstance(approach, str):
                    plan_parts.append(f"Plan approach: {approach}")
                tasks = phase.output_structured.get("tasks")
                if tasks and isinstance(tasks, list):
                    for task_item in tasks:
                        if isinstance(task_item, dict):
                            description = task_item.get("description", "")
                            task_files = task_item.get("files", [])
                            if description:
                                task_line = f"Task: {description}"
                                if task_files and isinstance(task_files, list):
                                    task_line += f" (files: {', '.join(str(filepath) for filepath in task_files)})"
                                plan_parts.append(task_line)
                if plan_parts:
                    context_chunks.append(redact_sensitive("\n".join(plan_parts)))

            elif phase.name == "implement" and phase.output_structured:
                impl_parts: list[str] = []
                files_changed = phase.output_structured.get("files_changed")
                if files_changed and isinstance(files_changed, list):
                    impl_parts.append(
                        f"Files changed: {', '.join(str(filepath) for filepath in files_changed)}"
                    )
                commits = phase.output_structured.get("commits")
                if commits and isinstance(commits, list):
                    commit_messages = []
                    for commit_item in commits:
                        if isinstance(commit_item, dict):
                            msg = commit_item.get("message", "")
                            if msg:
                                commit_messages.append(str(msg))
                    if commit_messages:
                        impl_parts.append(f"Commits: {'; '.join(commit_messages)}")
                deviations = phase.output_structured.get("deviations")
                if deviations and isinstance(deviations, str):
                    impl_parts.append(f"Deviations: {deviations}")
                if impl_parts:
                    context_chunks.append(redact_sensitive("\n".join(impl_parts)))

        # Fall back to implement phase output when structured fields are insufficient
        if not context_chunks:
            for phase in phases:
                if phase.name == "implement" and phase.output:
                    fallback_text = phase.output[:MAX_IMPLEMENT_FALLBACK_CHARS]
                    context_chunks.append(redact_sensitive(fallback_text))
                    break

        if not context_chunks:
            return None

        joined = "\n---\n".join(context_chunks)
        if len(joined) > MAX_SYNTHESIZED_CONTEXT_CHARS:
            joined = joined[:MAX_SYNTHESIZED_CONTEXT_CHARS]
        return joined

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
