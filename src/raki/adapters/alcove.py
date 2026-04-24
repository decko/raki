import json
from datetime import datetime, timezone
from pathlib import Path

from raki.adapters.redact import redact_dict, redact_sensitive
from raki.model import EvalSample, PhaseResult, ReviewFinding, SessionMeta, ToolCall

MAX_ALCOVE_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
DETECT_READ_SIZE = 4096  # Only read first 4KB for format detection
MAX_SYNTHESIZED_CONTEXT_CHARS = 50_000

# Bash commands whose output is informational for retrieval context (whitelist).
# Only include outputs from commands that start with one of these prefixes.
_INFORMATIONAL_BASH_PREFIXES = (
    "pytest",
    "uv run pytest",
    "uv run ruff",
    "uv run ty",
    "ruff",
    "make",
    "npm test",
    "cargo test",
    "grep",
    "rg",
    "find",
    "cat",
)

# Tool names classified by phase category for multi-phase detection.
_ANALYSIS_TOOLS = frozenset({"Read", "Grep", "Glob"})
_IMPLEMENTATION_TOOLS = frozenset({"Write", "Edit"})

# Bash commands that indicate a "testing" phase (test/lint runners).
_TEST_BASH_PREFIXES = (
    "pytest",
    "uv run pytest",
    "uv run ruff",
    "uv run ty",
    "ruff",
    "make test",
    "npm test",
    "cargo test",
    "go test",
)


def _classify_tool_call(tool_name: str, tool_input: dict | None) -> str | None:
    """Classify a tool call as 'analysis', 'coding', or 'testing'.

    Returns None for tool calls that don't map to a recognized phase.
    """
    if tool_name in _ANALYSIS_TOOLS:
        return "analysis"
    if tool_name in _IMPLEMENTATION_TOOLS:
        return "coding"
    if tool_name == "Bash" and isinstance(tool_input, dict):
        command = tool_input.get("command", "").lstrip()
        if command.startswith(_TEST_BASH_PREFIXES):
            return "testing"
    return None


def _is_test_failure(tool_result_content: str) -> bool:
    """Heuristic: does the tool_result content indicate a test/lint failure?"""
    failure_markers = ("FAILED", "ERRORS", "error:", "Error:", "FAIL")
    return any(marker in tool_result_content for marker in failure_markers)


def _extract_tool_sequence(transcript: list[dict]) -> list[dict]:
    """Walk the transcript and produce a sequence of tool-call records.

    Each record contains:
      - ``tool_name``: e.g. "Read", "Edit", "Bash"
      - ``tool_input``: the input dict (for Bash commands)
      - ``phase``: classified phase ("analysis"/"coding"/"testing") or None
      - ``result_content``: the tool_result text, if available
      - ``is_failure``: True if the result indicates a test failure
    """
    # Build maps from tool_use blocks
    tool_name_by_id: dict[str, str] = {}
    tool_input_by_id: dict[str, dict | None] = {}
    tool_order: list[str] = []  # ordered list of tool_use_ids

    for entry in transcript:
        if entry.get("type") != "assistant":
            continue
        message = entry.get("message", {})
        for content_block in message.get("content", []):
            if content_block.get("type") == "tool_use":
                tool_id = content_block.get("id", "")
                tool_name_by_id[tool_id] = content_block.get("name", "")
                raw_input = content_block.get("input")
                tool_input_by_id[tool_id] = raw_input if isinstance(raw_input, dict) else None
                tool_order.append(tool_id)

    # Collect tool results
    result_by_id: dict[str, str] = {}
    for entry in transcript:
        if entry.get("type") != "user":
            continue
        message = entry.get("message", {})
        for content_block in message.get("content", []):
            if not isinstance(content_block, dict):
                continue
            if content_block.get("type") == "tool_result":
                tool_id = content_block.get("tool_use_id", "")
                result_text = content_block.get("content", "")
                if isinstance(result_text, str):
                    result_by_id[tool_id] = result_text

    # Build the sequence
    sequence: list[dict] = []
    for tool_id in tool_order:
        tool_name = tool_name_by_id.get(tool_id, "")
        tool_input = tool_input_by_id.get(tool_id)
        phase = _classify_tool_call(tool_name, tool_input)
        result_content = result_by_id.get(tool_id, "")
        sequence.append(
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "phase": phase,
                "result_content": result_content,
                "is_failure": phase == "testing" and _is_test_failure(result_content),
            }
        )
    return sequence


def _detect_rework_cycles(tool_sequence: list[dict]) -> int:
    """Count rework cycles from a tool-call sequence.

    A rework cycle is: a test/lint command fails, then files are edited
    (implementation tools), then another test/lint command runs.

    TDD workflow (test written first, fails, then implementation, then pass)
    counts as 0 rework because the initial test failure precedes any editing
    of *previously written* files.  We track which files were modified;
    a rework cycle only counts when editing happens *after* a test failure
    on files that the agent already touched earlier in the session.
    """
    rework_count = 0
    files_written_before_failure: set[str] = set()
    awaiting_rework_edit = False
    rework_edit_seen = False

    for record in tool_sequence:
        phase = record["phase"]
        tool_input = record["tool_input"] or {}

        if phase == "coding":
            # Track which files have been written/edited
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            if file_path:
                if awaiting_rework_edit and file_path in files_written_before_failure:
                    rework_edit_seen = True
                files_written_before_failure.add(file_path)

        elif phase == "testing":
            # If we saw a rework edit since last failure, this test run
            # (pass or fail) completes one rework cycle.
            if rework_edit_seen:
                rework_count += 1
                rework_edit_seen = False

            if record["is_failure"]:
                # Test failed -- start (or continue) watching for rework edits
                awaiting_rework_edit = True
            else:
                # Test passed -- reset the rework state
                awaiting_rework_edit = False

    return rework_count


def _detect_phase_count(tool_sequence: list[dict]) -> int:
    """Count distinct phases from a tool-call sequence.

    Phases are contiguous groups of tool calls with the same classification.
    Transitions between 'analysis', 'coding', and 'testing' increment the count.
    Tool calls with no classification (phase=None) are ignored.

    Returns at least 1 (a session always has at least one phase).
    """
    if not tool_sequence:
        return 1

    phase_count = 0
    previous_phase: str | None = None

    for record in tool_sequence:
        phase = record["phase"]
        if phase is None:
            continue
        if phase != previous_phase:
            phase_count += 1
            previous_phase = phase

    return max(phase_count, 1)


def _extract_session_id(raw: dict) -> str:
    """Extract session ID from either alcove or bridge format.

    Bridge format uses ``id`` at the top level.
    Classic alcove format uses ``session_id``.
    Falls back to ``task_id`` or the system entry's ``session_id``.
    """
    if "session_id" in raw:
        return raw["session_id"]
    if "id" in raw:
        return raw["id"]
    if "task_id" in raw:
        return raw["task_id"]
    for entry in raw.get("transcript", []):
        if entry.get("type") == "system" and "session_id" in entry:
            return entry["session_id"]
    raise KeyError("No session_id or id found in transcript file")


class AlcoveAdapter:
    name: str = "alcove"
    description: str = "Single-file JSON transcript from Claude Code or Alcove bridge"
    detection_hint: str = "*.json file containing (session_id or id) + transcript"

    def detect(self, source: Path) -> bool:
        """Detect Alcove/bridge format via substring search of first 4KB."""
        if source.is_symlink():
            return False
        if not source.is_file() or source.suffix != ".json":
            return False
        try:
            with source.open(encoding="utf-8", errors="replace") as file_handle:
                header = file_handle.read(DETECT_READ_SIZE)
            has_transcript = '"transcript"' in header
            has_session_id = '"session_id"' in header
            has_bridge_id = '"id"' in header and '"task_id"' in header
            return has_transcript and (has_session_id or has_bridge_id)
        except OSError:
            return False

    def load(self, source: Path) -> EvalSample:
        """Parse a Claude Code or bridge transcript into an EvalSample."""
        if source.is_symlink():
            raise ValueError(f"Source must be a regular file (no symlinks): {source}")
        resolved = source.resolve()
        if not resolved.is_file():
            raise ValueError(f"Source must be a regular file: {source}")
        file_size = resolved.stat().st_size
        if file_size > MAX_ALCOVE_FILE_SIZE:
            raise ValueError(
                f"File exceeds {MAX_ALCOVE_FILE_SIZE // (1024 * 1024)}MB limit: {source}"
            )
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        transcript = raw["transcript"]
        is_bridge = "task_id" in raw

        session_id = _extract_session_id(raw)

        model_id: str | None = None
        tool_calls: list[ToolCall] = []
        output_parts: list[str] = []
        total_cost_usd: float | None = None
        duration_ms: int | None = None
        started_at: datetime | None = None
        tokens_in = 0
        tokens_out = 0
        session_status = raw.get("status", "completed") if is_bridge else "completed"
        task_name: str | None = raw.get("task_name") if is_bridge else None

        if is_bridge and raw.get("started_at"):
            started_at = datetime.fromisoformat(raw["started_at"])

        for entry in transcript:
            entry_type = entry.get("type")

            if entry_type == "system":
                model_id = entry.get("model")

            elif entry_type == "assistant":
                message = entry.get("message", {})
                usage = message.get("usage", {})
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
                for content_block in message.get("content", []):
                    if content_block.get("type") == "tool_use":
                        raw_args = content_block.get("input")
                        tool_calls.append(
                            ToolCall(
                                name=content_block["name"],
                                arguments=redact_dict(raw_args)
                                if isinstance(raw_args, dict)
                                else raw_args,
                            )
                        )
                    elif content_block.get("type") == "text":
                        output_parts.append(content_block["text"])

            elif entry_type == "user":
                timestamp_str = entry.get("timestamp")
                if timestamp_str and started_at is None:
                    started_at = datetime.fromisoformat(timestamp_str)

            elif entry_type == "result":
                total_cost_usd = entry.get("total_cost_usd")
                duration_ms = entry.get("duration_ms")
                model_usage = entry.get("modelUsage", {})
                if not model_id and model_usage:
                    model_id = next(iter(model_usage), None)

        phases_dict: dict = raw.get("phases") or {}
        findings = self._parse_findings(raw.get("findings") or [])

        output = redact_sensitive("\n".join(output_parts))
        phases = self._build_phases(
            phases_dict=phases_dict,
            output=output,
            tool_calls=tool_calls,
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            total_cost_usd=total_cost_usd,
            duration_ms=duration_ms,
            session_status=session_status,
        )

        # Detect rework cycles and phases from transcript tool calls.
        # Explicit values in the JSON take priority; transcript analysis is the fallback.
        tool_sequence = _extract_tool_sequence(transcript)

        if "rework_cycles" in raw:
            rework_cycles = raw["rework_cycles"]
        else:
            rework_cycles = _detect_rework_cycles(tool_sequence)

        if phases_dict:
            total_phases = len(phases_dict)
        else:
            total_phases = _detect_phase_count(tool_sequence)

        ticket = task_name or session_id
        meta = SessionMeta(
            session_id=session_id,
            ticket=ticket if ticket != session_id else None,
            started_at=started_at or datetime.now(timezone.utc),
            total_cost_usd=total_cost_usd,
            total_phases=total_phases,
            rework_cycles=rework_cycles,
            model_id=model_id,
        )

        sample = EvalSample(
            session=meta,
            phases=phases,
            findings=findings,
            events=[],
        )

        # Synthesize context from tool outputs if no phase has explicit knowledge_context
        has_explicit_context = any(phase.knowledge_context is not None for phase in sample.phases)
        if not has_explicit_context:
            synthesized = self._synthesize_context(transcript)
            if synthesized:
                sample.phases[0].knowledge_context = synthesized
                sample.context_source = "synthesized"

        return sample

    def _build_phases(
        self,
        phases_dict: dict,
        output: str,
        tool_calls: list[ToolCall],
        tokens_in: int | None,
        tokens_out: int | None,
        total_cost_usd: float | None,
        duration_ms: int | None,
        session_status: str,
    ) -> list[PhaseResult]:
        """Build PhaseResult list from phases dict or a single 'session' phase.

        When ``phases_dict`` is non-empty, creates one ``PhaseResult`` per
        phase entry.  The transcript-derived output and tool_calls are placed
        on the last phase (the primary output phase).  All other phases receive
        an empty output string and no tool calls.

        When ``phases_dict`` is empty or absent, falls back to a single phase
        named ``"session"`` containing all transcript data.
        """
        _VALID_STATUSES = {"completed", "failed", "skipped"}

        if not phases_dict:
            return [
                PhaseResult(
                    name="session",
                    generation=1,
                    status="completed" if session_status == "completed" else "failed",
                    cost_usd=total_cost_usd,
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    output=output,
                    tool_calls=tool_calls,
                )
            ]

        phase_entries = list(phases_dict.items())
        phases: list[PhaseResult] = []
        for idx, (phase_name, phase_meta) in enumerate(phase_entries):
            phase_meta = phase_meta or {}
            is_last = idx == len(phase_entries) - 1
            raw_status = phase_meta.get("status", "completed")
            status = raw_status if raw_status in _VALID_STATUSES else "completed"
            # Per-phase token counts from phases dict take priority over transcript totals
            phase_tokens_in = phase_meta.get("tokens_in")
            phase_tokens_out = phase_meta.get("tokens_out")
            phases.append(
                PhaseResult(
                    name=phase_name,
                    generation=phase_meta.get("generation", 1),
                    status=status,
                    cost_usd=phase_meta.get("cost_usd", phase_meta.get("cost")),
                    duration_ms=phase_meta.get("duration_ms"),
                    tokens_in=phase_tokens_in
                    if phase_tokens_in is not None
                    else (tokens_in if is_last else None),
                    tokens_out=phase_tokens_out
                    if phase_tokens_out is not None
                    else (tokens_out if is_last else None),
                    output=output if is_last else "",
                    tool_calls=tool_calls if is_last else [],
                )
            )
        return phases

    def _parse_findings(self, raw_findings: list) -> list[ReviewFinding]:
        """Parse a list of raw finding dicts into ``ReviewFinding`` objects.

        Skips malformed entries (missing required ``issue`` key or invalid
        severity).  Applies ``redact_sensitive()`` to free-text fields.
        """
        findings: list[ReviewFinding] = []
        for finding_raw in raw_findings:
            if not isinstance(finding_raw, dict):
                continue
            try:
                findings.append(
                    ReviewFinding(
                        reviewer=finding_raw.get("source", "unknown"),
                        severity=finding_raw.get("severity", "minor"),
                        file=finding_raw.get("file"),
                        line=finding_raw.get("line"),
                        issue=redact_sensitive(finding_raw["issue"]),
                        suggestion=redact_sensitive(suggestion)
                        if (suggestion := finding_raw.get("suggestion")) is not None
                        else None,
                    )
                )
            except (KeyError, ValueError):
                continue
        return findings

    def _synthesize_context(self, transcript: list[dict]) -> str | None:
        """Extract retrieval context from tool call outputs in the transcript.

        Iterates through the transcript looking for tool_use blocks followed by
        tool_result responses. Extracts content from Read, Grep, and informational
        Bash tool calls.
        """
        # Build a map of tool_use_id -> tool_name from assistant messages
        tool_name_by_id: dict[str, str] = {}
        tool_command_by_id: dict[str, str] = {}

        for entry in transcript:
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message", {})
            for content_block in message.get("content", []):
                if content_block.get("type") == "tool_use":
                    tool_id = content_block.get("id", "")
                    tool_name = content_block.get("name", "")
                    tool_name_by_id[tool_id] = tool_name
                    tool_input = content_block.get("input", {})
                    if isinstance(tool_input, dict):
                        tool_command_by_id[tool_id] = tool_input.get("command", "")

        # Extract tool result content
        context_chunks: list[str] = []

        for entry in transcript:
            if entry.get("type") != "user":
                continue
            message = entry.get("message", {})
            for content_block in message.get("content", []):
                if not isinstance(content_block, dict):
                    continue
                if content_block.get("type") != "tool_result":
                    continue

                tool_id = content_block.get("tool_use_id", "")
                tool_name = tool_name_by_id.get(tool_id, "")
                result_content = content_block.get("content", "")

                if not result_content or not isinstance(result_content, str):
                    continue

                if tool_name in ("Read", "Grep"):
                    redacted_chunk = redact_sensitive(result_content)
                    context_chunks.append(redacted_chunk)
                elif tool_name == "Bash":
                    command = tool_command_by_id.get(tool_id, "").lstrip()
                    if not command.startswith(_INFORMATIONAL_BASH_PREFIXES):
                        continue
                    redacted_chunk = redact_sensitive(result_content)
                    context_chunks.append(redacted_chunk)

        if not context_chunks:
            return None

        joined = "\n---\n".join(context_chunks)
        if len(joined) > MAX_SYNTHESIZED_CONTEXT_CHARS:
            joined = joined[:MAX_SYNTHESIZED_CONTEXT_CHARS]
        return joined
