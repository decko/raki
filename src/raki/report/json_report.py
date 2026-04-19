"""JSON report serialization — write/load round-trip with optional session stripping."""

import json
from pathlib import Path

from raki.model.report import EvalReport


def write_json_report(report: EvalReport, output: Path, include_sessions: bool = False) -> None:
    """Write JSON report. By default strips large session data fields to keep reports compact.

    Stripped fields (when include_sessions=False):
    - PhaseResult.output (can be very large for implement phases)
    - ToolCall.arguments (may contain sensitive data or large payloads)
    - SessionEvent.data (raw event payloads)

    Use include_sessions=True (or --include-sessions CLI flag) to retain full data.
    """
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = report.model_dump(mode="json")
    if not include_sessions:
        strip_session_data(data)
    output.write_text(json.dumps(data, indent=2, default=str))


def strip_session_data(data: dict) -> None:
    """Remove large raw data fields from sample results to keep reports compact."""
    for sample_result in data.get("sample_results", []):
        sample = sample_result.get("sample", {})
        for phase in sample.get("phases", []):
            # output is a required str field — replace with sentinel instead of removing
            phase["output"] = "<stripped>"
            # optional fields — safe to remove entirely
            phase.pop("output_structured", None)
            phase.pop("knowledge_context", None)
            phase.pop("instruction_context", None)
            for tool_call in phase.get("tool_calls", []):
                tool_call.pop("arguments", None)
        for event in sample.get("events", []):
            event.pop("data", None)


def load_json_report(path: Path) -> EvalReport:
    """Load an EvalReport from a JSON file."""
    raw = json.loads(path.read_text())
    return EvalReport.model_validate(raw)


def timestamp_filename(report: EvalReport) -> str:
    """Generate a timestamp-based filename from the report's timestamp.

    Uses full datetime (not date-only) to avoid overwrites when multiple
    evaluations run on the same day.
    """
    timestamp = report.timestamp
    formatted = timestamp.strftime("%Y%m%dT%H%M%S")
    return f"raki-report-{formatted}.json"
