"""Alcove pipeline-run export adapter.

Detects and loads a directory produced by an Alcove pipeline export — i.e. a
folder containing ``run.json`` (with step-level metadata) and a ``steps/``
sub-directory where each step's transcript or bridge metadata is stored.

Format outline::

    <export-dir>/
        run.json          # pipeline-run manifest (id, steps[], …)
        steps/
            01-triage/
                transcript.json   # agent steps
            02-plan/
                transcript.json
            07-create-pr/
                step.json         # bridge / skipped steps
            …

Each agent step directory contains a ``transcript.json`` in the standard
Alcove transcript format.  Bridge and skipped steps contain a lighter-weight
``step.json`` with only the step metadata.

Mapping rules
-------------
- Every step → one :class:`PhaseResult` whose ``name`` is the ``step_id``.
- Phase status: ``"skipped"`` → ``"skipped"``; ``"completed"`` → ``"completed"``;
  verify step with ``outputs.verdict == "fail"`` (case-insensitive) → ``"failed"``.
- Review findings: parsed from semicolon-delimited ``outputs.issues`` on
  review-type steps.  Each token starts with ``CRITICAL:``, ``MAJOR:``, or
  ``MINOR:`` (case-insensitive).
- Rework cycles: count of non-skipped *corrective* agent steps (those whose
  ``depends`` condition references a ``*.Failed`` event, e.g. ``patch``,
  ``revision``, ``ci-fix``).
- Total cost: sum of ``total_cost_usd`` across all loaded transcripts.
- Session ID: the ``run_id`` shared by all steps in ``run.json``.
- Started at: earliest ``started_at`` across non-skipped steps.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from raki.adapters.alcove import parse_transcript
from raki.adapters.redact import redact_dict, redact_sensitive
from raki.model import EvalSample, PhaseResult, ReviewFinding, SessionMeta, ToolCall

# Read only the first 4 KB of run.json for format detection.
_DETECT_READ_SIZE = 4096

# Corrective step IDs — steps that are activated on verify/review failure.
# A non-skipped corrective step counts as one rework cycle.
_CORRECTIVE_STEP_IDS: frozenset[str] = frozenset({"patch", "revision", "ci-fix"})

# Severity prefix map (upper-case keys).
_SEVERITY_MAP: dict[str, Literal["critical", "major", "minor"]] = {
    "CRITICAL": "critical",
    "MAJOR": "major",
    "MINOR": "minor",
}

# Maximum characters of synthesized context to store on the implement phase.
_MAX_CONTEXT_CHARS = 50_000


def _parse_duration_ms(started_at: str | None, finished_at: str | None) -> int | None:
    """Compute duration in milliseconds from ISO-format timestamp strings.

    Returns ``None`` if either timestamp is absent or unparseable.
    """
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        finish = datetime.fromisoformat(finished_at)
        delta_ms = int((finish - start).total_seconds() * 1000)
        return delta_ms if delta_ms >= 0 else None
    except (ValueError, TypeError):
        return None


def _parse_issues(issues_text: str, reviewer: str) -> list[ReviewFinding]:
    """Parse semicolon-delimited review issues into :class:`ReviewFinding` objects.

    Each token is expected to start with a severity prefix followed by a colon
    and the issue description, e.g.::

        MAJOR: <description>; MINOR: <description>

    Tokens that lack a recognised severity prefix are silently skipped.
    Severity matching is case-insensitive.

    Args:
        issues_text: The raw ``outputs.issues`` string from a review step.
        reviewer: The ``step_id`` of the review step (used as the reviewer name).

    Returns:
        A list of :class:`ReviewFinding` objects, one per valid token.
    """
    findings: list[ReviewFinding] = []
    tokens = [token.strip() for token in issues_text.split(";") if token.strip()]
    for token in tokens:
        # Find the first colon and treat the prefix as the severity label.
        colon_idx = token.find(":")
        if colon_idx <= 0:
            continue
        prefix = token[:colon_idx].strip().upper()
        severity = _SEVERITY_MAP.get(prefix)
        if severity is None:
            continue
        description = token[colon_idx + 1 :].strip()
        if not description:
            continue
        findings.append(
            ReviewFinding(
                reviewer=reviewer,
                severity=severity,
                issue=redact_sensitive(description),
                finding_source="review",
            )
        )
    return findings


def _is_corrective(step_depends: str | None) -> bool:
    """Return True when a step's ``depends`` condition references a ``*.Failed`` event.

    Corrective steps are agent steps that are only activated after a previous
    step fails, such as ``patch`` (``depends: "verify.Failed"``) or ``revision``
    (``depends: "review-django.Failed || review-security.Failed"``).
    """
    if not step_depends:
        return False
    return bool(re.search(r"\w+\.Failed", step_depends))


def _build_phase_name(step_id: str) -> str:
    """Return the canonical phase name for a given ``step_id``.

    The phase name is kept equal to the ``step_id`` so that per-step
    granularity is preserved.  Consumer code (metrics, reports) must not
    assume SODA-style names (``triage``, ``plan``, …) for pipeline sessions.
    """
    return step_id


def _phase_status(
    step_status: str,
    step_id: str,
    outputs: dict,
) -> Literal["completed", "failed", "skipped"]:
    """Map an Alcove pipeline step status to a :class:`PhaseResult` status literal.

    Rules:
    - ``"skipped"`` → ``"skipped"``
    - verify step with ``outputs.verdict`` == ``"fail"`` (case-insensitive) → ``"failed"``
    - everything else ``"completed"`` → ``"completed"``
    """
    if step_status == "skipped":
        return "skipped"
    if step_id == "verify" or step_id.startswith("verify"):
        verdict = outputs.get("verdict") if outputs else None
        if isinstance(verdict, str) and verdict.strip().upper() == "FAIL":
            return "failed"
    if step_status == "failed":
        return "failed"
    return "completed"


class AlcovePipelineAdapter:
    """Adapter for Alcove pipeline-run export directories.

    A pipeline export directory contains a top-level ``run.json`` (with
    step metadata) and a ``steps/`` subdirectory with per-step transcripts
    or step-metadata files.
    """

    name: str = "alcove-pipeline"
    description: str = "Alcove pipeline-run export (run.json + steps/ directory)"
    detection_hint: str = "directory with run.json containing a steps array"

    def detect(self, source: Path) -> bool:
        """Return True when *source* looks like an Alcove pipeline export directory.

        Checks:
        1. *source* is a non-symlink directory.
        2. A ``run.json`` file exists at the top level.
        3. The first 4 KB of ``run.json`` contains ``"steps"``.
        """
        if source.is_symlink():
            return False
        if not source.is_dir():
            return False
        run_json = source / "run.json"
        if not run_json.is_file():
            return False
        try:
            with run_json.open(encoding="utf-8", errors="replace") as file_handle:
                header = file_handle.read(_DETECT_READ_SIZE)
            return '"steps"' in header
        except OSError:
            return False

    def load(self, source: Path) -> EvalSample:
        """Load an Alcove pipeline export directory into an :class:`EvalSample`.

        Args:
            source: Path to the pipeline export directory.

        Returns:
            A fully-populated :class:`EvalSample`.

        Raises:
            ValueError: If *source* is a symlink or the ``run.json`` is malformed.
        """
        if source.is_symlink():
            raise ValueError(f"Refusing to load symlink: {source}")

        run_json_path = source / "run.json"
        run_raw: dict = json.loads(run_json_path.read_text(encoding="utf-8"))
        steps_raw: list[dict] = run_raw.get("steps") or []

        # Build a lookup from step_id → list of step dicts (handles multiple iterations).
        steps_by_id: dict[str, list[dict]] = {}
        for step in steps_raw:
            step_id = step.get("step_id", "")
            steps_by_id.setdefault(step_id, []).append(step)

        # Determine run_id from any step that carries it.
        run_id: str = ""
        for step in steps_raw:
            candidate = step.get("run_id", "")
            if candidate:
                run_id = candidate
                break

        # Discover the steps/ sub-directory and enumerate ordered step directories.
        steps_dir = source / "steps"
        ordered_step_dirs = self._ordered_step_dirs(steps_dir)

        phases: list[PhaseResult] = []
        findings: list[ReviewFinding] = []
        total_cost_usd: float = 0.0
        has_any_cost = False
        started_at: datetime | None = None
        model_id: str | None = None
        all_tool_calls: list[ToolCall] = []
        implement_output: str = ""
        context_chunks: list[str] = []

        for step_dir in ordered_step_dirs:
            step_id = self._step_id_from_dir(step_dir)
            step_meta = self._pick_step_meta(steps_by_id, step_id)
            step_status = step_meta.get("status", "completed") if step_meta else "completed"
            step_outputs: dict = (step_meta.get("outputs") or {}) if step_meta else {}
            step_started: str | None = step_meta.get("started_at") if step_meta else None
            step_finished: str | None = step_meta.get("finished_at") if step_meta else None
            step_iteration: int = int(step_meta.get("iteration", 1)) if step_meta else 1

            # Track pipeline start time.
            if step_started:
                parsed_start = self._parse_timestamp(step_started)
                if parsed_start is not None:
                    if started_at is None or parsed_start < started_at:
                        started_at = parsed_start

            duration_ms = _parse_duration_ms(step_started, step_finished)
            phase_status = _phase_status(step_status, step_id, step_outputs)

            # Load transcript (agent steps) or use step metadata only.
            transcript_path = step_dir / "transcript.json"
            cost_usd: float | None = None
            tokens_in: int | None = None
            tokens_out: int | None = None
            output_text = ""
            step_tool_calls: list[ToolCall] = []

            if transcript_path.is_file():
                transcript_raw = json.loads(transcript_path.read_text(encoding="utf-8"))
                td = parse_transcript(transcript_raw)
                if td.total_cost_usd is not None:
                    cost_usd = td.total_cost_usd
                    total_cost_usd += td.total_cost_usd
                    has_any_cost = True
                if td.duration_ms is not None and duration_ms is None:
                    duration_ms = td.duration_ms
                tokens_in = td.tokens_in or None
                tokens_out = td.tokens_out or None
                output_text = td.output
                step_tool_calls = td.tool_calls
                if model_id is None and td.model_id:
                    model_id = td.model_id
                all_tool_calls.extend(step_tool_calls)

                # Collect context chunks from triage and plan steps.
                if step_id in ("triage", "plan"):
                    chunk = self._context_chunk_from_outputs(step_id, step_outputs)
                    if chunk:
                        context_chunks.append(chunk)

                # Preserve implement output for context fallback.
                if step_id == "implement" and output_text:
                    implement_output = output_text

            # Parse review findings from outputs.issues.
            issues_text = step_outputs.get("issues", "") if step_outputs else ""
            if isinstance(issues_text, str) and issues_text.strip():
                findings.extend(_parse_issues(issues_text, reviewer=step_id))

            phase_name = _build_phase_name(step_id)
            phases.append(
                PhaseResult(
                    name=phase_name,
                    generation=step_iteration,
                    status=phase_status,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    output=output_text,
                    tool_calls=step_tool_calls,
                    output_structured=redact_dict(step_outputs) if step_outputs else None,
                )
            )

        # Count rework cycles: non-skipped corrective agent steps.
        rework_cycles = self._count_rework_cycles(steps_raw)

        # Build pipeline_phases from the ordered step ids.
        pipeline_phase_names = [self._step_id_from_dir(step_dir) for step_dir in ordered_step_dirs]

        meta = SessionMeta(
            session_id=run_id or source.name,
            ticket=None,
            started_at=started_at or datetime.now(timezone.utc),
            total_cost_usd=total_cost_usd if has_any_cost else None,
            total_phases=sum(1 for phase in phases if phase.status != "skipped"),
            rework_cycles=rework_cycles,
            model_id=model_id,
            orchestrator="alcove",
            pipeline_phases=pipeline_phase_names if pipeline_phase_names else None,
        )

        sample = EvalSample(
            session=meta,
            phases=phases,
            findings=findings,
            events=[],
        )

        # Attach synthesized context to the implement phase (or last phase).
        if not any(phase.knowledge_context is not None for phase in phases):
            synthesized = self._synthesize_context(context_chunks, implement_output)
            if synthesized:
                target = self._find_context_target(phases)
                if target is not None:
                    target.knowledge_context = synthesized
                    sample.context_source = "synthesized"

        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ordered_step_dirs(self, steps_dir: Path) -> list[Path]:
        """Return step sub-directories from *steps_dir* sorted by numeric prefix.

        Expects directory names like ``01-triage``, ``02-plan``, …  Directories
        without a numeric prefix are placed at the end in alphabetical order.
        """
        if not steps_dir.is_dir():
            return []

        def _sort_key(path: Path) -> tuple[int, str]:
            match = re.match(r"^(\d+)-", path.name)
            if match:
                return (int(match.group(1)), path.name)
            return (999999, path.name)

        return sorted(
            (child for child in steps_dir.iterdir() if child.is_dir() and not child.is_symlink()),
            key=_sort_key,
        )

    def _step_id_from_dir(self, step_dir: Path) -> str:
        """Extract the step_id from a directory name like ``01-triage`` → ``"triage"``."""
        match = re.match(r"^\d+-(.+)$", step_dir.name)
        if match:
            return match.group(1)
        return step_dir.name

    def _pick_step_meta(self, steps_by_id: dict[str, list[dict]], step_id: str) -> dict | None:
        """Return the best matching step metadata dict for *step_id*.

        When there are multiple iterations of the same step, returns the one
        with the highest iteration number (most recent).  Returns ``None`` when
        no matching step exists.
        """
        candidates = steps_by_id.get(step_id)
        if not candidates:
            return None
        return max(candidates, key=lambda step: int(step.get("iteration", 0)))

    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        """Parse an ISO-format timestamp string; return ``None`` on failure."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            return None

    def _count_rework_cycles(self, steps_raw: list[dict]) -> int:
        """Count rework cycles from the pipeline step list.

        A rework cycle is one non-skipped corrective agent step.  Corrective
        steps are those whose ``depends`` condition references a ``*.Failed``
        event (e.g. ``verify.Failed``, ``review-django.Failed``).
        """
        count = 0
        for step in steps_raw:
            if step.get("status") == "skipped":
                continue
            if step.get("type") != "agent":
                continue
            if _is_corrective(step.get("depends")):
                count += 1
        return count

    def _context_chunk_from_outputs(self, step_id: str, outputs: dict) -> str | None:
        """Build a context text chunk from a step's structured outputs.

        Extracts the most useful fields from triage (approach, candidate_files,
        risks) and plan (plan text) outputs for retrieval context.
        """
        if not outputs:
            return None
        parts: list[str] = []
        if step_id == "triage":
            approach = outputs.get("approach")
            if approach and isinstance(approach, str):
                parts.append(f"Triage approach: {approach[:2000]}")
            candidate_files = outputs.get("candidate_files")
            if candidate_files and isinstance(candidate_files, str):
                parts.append(f"Candidate files: {candidate_files}")
            risks = outputs.get("risks")
            if risks and isinstance(risks, str):
                parts.append(f"Risks: {risks[:500]}")
        elif step_id == "plan":
            plan_text = outputs.get("plan")
            if plan_text and isinstance(plan_text, str):
                parts.append(f"Plan: {plan_text[:3000]}")
        if not parts:
            return None
        return redact_sensitive("\n".join(parts))

    def _synthesize_context(self, context_chunks: list[str], implement_output: str) -> str | None:
        """Assemble a knowledge context string from collected context chunks.

        Falls back to the implement step output when no structured chunks were
        collected.  Truncates to ``_MAX_CONTEXT_CHARS``.
        """
        if not context_chunks and not implement_output:
            return None
        if context_chunks:
            joined = "\n---\n".join(context_chunks)
        else:
            joined = implement_output[:10_000]
        if len(joined) > _MAX_CONTEXT_CHARS:
            joined = joined[:_MAX_CONTEXT_CHARS]
        return joined

    def _find_context_target(self, phases: list[PhaseResult]) -> PhaseResult | None:
        """Return the phase on which to attach synthesized knowledge context.

        Prefers the ``implement`` phase; falls back to the last non-skipped phase.
        """
        for phase in reversed(phases):
            if phase.name == "implement" and phase.status != "skipped":
                return phase
        for phase in reversed(phases):
            if phase.status != "skipped":
                return phase
        return phases[-1] if phases else None
