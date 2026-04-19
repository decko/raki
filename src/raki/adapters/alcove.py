import json
from datetime import datetime, timezone
from pathlib import Path

from raki.adapters.redact import redact_dict, redact_sensitive
from raki.model import EvalSample, PhaseResult, SessionMeta, ToolCall

MAX_ALCOVE_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
DETECT_READ_SIZE = 4096  # Only read first 4KB for format detection


class AlcoveAdapter:
    name: str = "alcove"
    description: str = "Single-file JSON transcript with session_id and transcript array"
    detection_hint: str = "*.json file containing session_id + transcript"

    def detect(self, source: Path) -> bool:
        """Detect Alcove format via substring search of first 4KB."""
        if not source.is_file() or source.suffix != ".json":
            return False
        try:
            with source.open(errors="replace") as file_handle:
                header = file_handle.read(DETECT_READ_SIZE)
            return '"session_id"' in header and '"transcript"' in header
        except OSError:
            return False

    def load(self, source: Path) -> EvalSample:
        """Parse an Alcove single-file transcript into an EvalSample."""
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
        session_id = raw["session_id"]
        transcript = raw["transcript"]

        model_id: str | None = None
        tool_calls: list[ToolCall] = []
        output_parts: list[str] = []
        total_cost_usd: float | None = None
        duration_ms: int | None = None
        started_at: datetime | None = None
        tokens_in = 0
        tokens_out = 0

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

        meta = SessionMeta(
            session_id=session_id,
            started_at=started_at or datetime.now(timezone.utc),
            total_cost_usd=total_cost_usd,
            total_phases=1,
            rework_cycles=0,
            model_id=model_id,
        )

        output = redact_sensitive("\n".join(output_parts))
        phase = PhaseResult(
            name="session",
            generation=1,
            status="completed",
            cost_usd=total_cost_usd,
            duration_ms=duration_ms,
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            output=output,
            tool_calls=tool_calls,
        )

        return EvalSample(
            session=meta,
            phases=[phase],
            findings=[],
            events=[],
        )
