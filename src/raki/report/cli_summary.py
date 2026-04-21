"""Rich CLI summary output — color-coded metrics grouped by category."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.console import Console

from raki.metrics.protocol import Metric
from raki.model.report import EvalReport

if TYPE_CHECKING:
    from raki.report.diff import DiffReport, SessionTransition

OPERATIONAL_METRICS = {
    "first_pass_verify_rate",
    "rework_cycles",
    "review_severity_distribution",
    "cost_efficiency",
    "self_correction_rate",
    "phase_execution_time",
    "token_efficiency",
}

KNOWLEDGE_METRICS = {
    "knowledge_gap_rate",
    "knowledge_miss_rate",
}

EXPERIMENTAL_METRICS = {
    "faithfulness",
    "answer_relevancy",
}

CONTEXT_SENSITIVE_METRICS = {
    "faithfulness",
    "answer_relevancy",
}


class _MetricMeta:
    """Lookup helper for metric display metadata."""

    def __init__(self, metrics: Sequence[Metric] | None = None) -> None:
        self._by_name: dict[str, Metric] = {}
        if metrics:
            for metric in metrics:
                self._by_name[metric.name] = metric

    def display_name(self, name: str) -> str:
        metric = self._by_name.get(name)
        return metric.display_name if metric else name

    def description(self, name: str) -> str:
        metric = self._by_name.get(name)
        return metric.description if metric else ""

    def display_format(self, name: str) -> str:
        metric = self._by_name.get(name)
        return metric.display_format if metric else "score"

    def higher_is_better(self, name: str) -> bool:
        metric = self._by_name.get(name)
        return metric.higher_is_better if metric else True


def color_for_score(
    score: float, higher_is_better: bool = True, display_format: str = "score"
) -> str:
    """Color-code a score value.

    Skip color for non-ratio metrics (currency, count) where higher_is_better
    is False -- those values are not on a 0-1 scale.
    """
    if not higher_is_better and display_format in ("currency", "count", "duration"):
        return "white"
    if higher_is_better:
        if score >= 0.8:
            return "green"
        if score >= 0.6:
            return "yellow"
        return "red"
    else:
        # Inverted scale: lower is better (e.g., rework cycles as a ratio)
        if score <= 0.2:
            return "green"
        if score <= 0.4:
            return "yellow"
        return "red"


def _has_no_data(metric_details: dict[str, dict], metric_name: str) -> bool:
    """Check if a metric has no applicable data based on its details dict."""
    details = metric_details.get(metric_name, {})
    if "skipped" in details:
        return True
    for key, value in details.items():
        if key.startswith("sessions_with_") and value == 0:
            return True
    return False


def _is_synthesized_context(metric_details: dict[str, dict], metric_name: str) -> bool:
    """Check if a metric used synthesized (inferred) context."""
    details = metric_details.get(metric_name, {})
    return details.get("context_source") == "synthesized"


def _no_data_reason(metric_details: dict[str, dict], metric_name: str) -> str:
    """Return a human-readable reason why a metric has no data."""
    details = metric_details.get(metric_name, {})
    skipped = details.get("skipped")
    if skipped:
        return str(skipped)
    return "no data"


def format_metric_line(
    name: str,
    score: float | None,
    detail: str = "",
    display_format: str = "score",
    higher_is_better: bool = True,
    sample_count: int | None = None,
    display_name: str | None = None,
    no_data: bool = False,
    no_data_reason: str = "no data",
) -> str:
    """Format a single metric line with color, display format, and sample count."""
    label = display_name or name
    if no_data or score is None:
        reason = no_data_reason if no_data else "no applicable data"
        return f"[dim]  {label:<35} N/A    ({reason})[/dim]"
    color = color_for_score(score, higher_is_better, display_format)
    if display_format == "currency":
        score_str = f"${score:.2f}"
    elif display_format == "count":
        score_str = f"{score:.1f}"
    elif display_format == "percent":
        score_str = f"{score:.2f}"
    elif display_format == "duration":
        score_str = f"{score:.1f}s"
    else:
        score_str = f"{score:.2f}"
    count_str = f" (n={sample_count})" if sample_count is not None else ""
    detail_str = f"  ({detail})" if detail else ""
    return f"[{color}]  {label:<35} {score_str}{count_str}{detail_str}[/{color}]"


def generate_summary_sentence(report: EvalReport, session_count: int) -> str:
    """Generate a plain-English summary sentence from aggregate scores.

    The summary highlights key numbers: verify rate, rework cycles, finding counts,
    and average cost. Missing metrics are gracefully omitted.
    """
    parts: list[str] = []
    scores = report.aggregate_scores

    verify_rate = scores.get("first_pass_verify_rate")
    if verify_rate is not None:
        pct = f"{verify_rate * 100:.0f}%"
        parts.append(f"{pct} of sessions passed on first try")

    rework = scores.get("rework_cycles")
    if rework is not None:
        parts.append(f"with an average of {rework:.1f} rework cycles")

    # Count findings from sample_results
    severity_counter: Counter[str] = Counter()
    for sample_result in report.sample_results:
        for finding in sample_result.sample.findings:
            severity_counter[finding.severity] += 1

    if severity_counter:
        finding_parts: list[str] = []
        for severity_level in ("critical", "major", "minor"):
            count = severity_counter.get(severity_level, 0)
            if count > 0:
                finding_parts.append(f"{count} {severity_level}")
        if finding_parts:
            finding_text = ", ".join(finding_parts)
            parts.append(f"Reviewers found {finding_text} issues across all sessions")

    cost = scores.get("cost_efficiency")
    if cost is not None:
        parts.append(f"Average cost: ${cost:.2f}/session")

    if not parts:
        return f"Evaluation completed across {session_count} sessions."

    # Join the first two parts with comma, rest with period
    sentence_parts: list[str] = []
    if len(parts) >= 2 and "passed on first try" in parts[0]:
        sentence_parts.append(f"{parts[0]}, {parts[1]}")
        remaining = parts[2:]
    else:
        sentence_parts.append(parts[0])
        remaining = parts[1:]

    for part in remaining:
        sentence_parts.append(part)

    return ". ".join(sentence_parts) + "."


def print_summary(
    report: EvalReport,
    session_count: int,
    skipped_count: int = 0,
    error_count: int = 0,
    console: Console | None = None,
    metrics: Sequence[Metric] | None = None,
) -> None:
    """Print a Rich CLI summary of the evaluation report.

    Metrics are split into two categories: Operational Health and Retrieval Quality.
    A small sample caveat is shown alongside scores when n < 50.

    When *metrics* is provided, human-readable ``display_name`` and
    ``description`` values are shown instead of raw metric names.
    """
    output_console = console or Console()
    output_console.print()

    meta = _MetricMeta(metrics)

    non_retrieval = OPERATIONAL_METRICS | KNOWLEDGE_METRICS
    operational = {
        name: score for name, score in report.aggregate_scores.items() if name in non_retrieval
    }
    retrieval = {
        name: score for name, score in report.aggregate_scores.items() if name not in non_retrieval
    }

    if operational:
        output_console.print("[bold]Operational Health[/bold]")
        for name, score in operational.items():
            metric_no_data = _has_no_data(report.metric_details, name)
            output_console.print(
                format_metric_line(
                    name,
                    score,
                    detail=meta.description(name),
                    display_format=meta.display_format(name),
                    higher_is_better=meta.higher_is_better(name),
                    display_name=meta.display_name(name),
                    no_data=metric_no_data,
                    no_data_reason=_no_data_reason(report.metric_details, name)
                    if metric_no_data
                    else "no data",
                )
            )
        if session_count < 50:
            output_console.print(
                f"[dim]  \u26a0 Small sample size (n={session_count}) \u2014 "
                "scores are directional, not definitive[/dim]"
            )
        output_console.print()

    if retrieval:
        output_console.print("[bold]Retrieval Quality[/bold]")
        for name, score in retrieval.items():
            tag = " [yellow]\\[experimental][/yellow]" if name in EXPERIMENTAL_METRICS else ""
            metric_no_data = _has_no_data(report.metric_details, name)
            # Check if context was synthesized for this metric
            inferred_tag = ""
            if (
                not metric_no_data
                and name in CONTEXT_SENSITIVE_METRICS
                and _is_synthesized_context(report.metric_details, name)
            ):
                inferred_tag = " [cyan]\\(inferred)[/cyan]"
            output_console.print(
                format_metric_line(
                    name,
                    score,
                    detail=meta.description(name),
                    display_format=meta.display_format(name),
                    higher_is_better=meta.higher_is_better(name),
                    display_name=meta.display_name(name),
                    no_data=metric_no_data,
                    no_data_reason=_no_data_reason(report.metric_details, name)
                    if metric_no_data
                    else "no data",
                )
                + ("" if metric_no_data else tag + inferred_tag)
            )
        if session_count < 50:
            output_console.print(
                f"[dim]  \u26a0 Small sample size (n={session_count}) \u2014 "
                "scores are directional, not definitive[/dim]"
            )
        output_console.print(
            "[dim]  \u26a0 Scores produced by same-provider LLM judge "
            "\u2014 calibration caveat applies[/dim]"
        )
        output_console.print()

    if skipped_count > 0 or error_count > 0:
        output_console.print(
            f"[dim]  {session_count} evaluated, {skipped_count} skipped, {error_count} errors[/dim]"
        )


def _format_delta_value(value: float, display_format: str) -> str:
    """Format a metric value for the diff summary."""
    if display_format == "currency":
        return f"${value:.2f}"
    if display_format == "count":
        return f"{value:.1f}"
    if display_format == "percent":
        return f"{value * 100:.0f}%"
    if display_format == "duration":
        return f"{value:.1f}s"
    return f"{value:.2f}"


def _format_delta_change(delta: float, display_format: str) -> str:
    """Format a delta change value with sign."""
    sign = "+" if delta >= 0 else ""
    if display_format == "currency":
        if delta < 0:
            return f"-${abs(delta):.2f}"
        return f"{sign}${delta:.2f}"
    if display_format == "count":
        return f"{sign}{delta:.1f}"
    if display_format == "percent":
        return f"{sign}{delta * 100:.0f}%"
    if display_format == "duration":
        return f"{sign}{delta:.1f}s"
    return f"{sign}{delta:.2f}"


def print_diff_summary(
    diff: DiffReport,
    console: Console | None = None,
) -> None:
    """Print a Rich CLI diff summary comparing two evaluation runs.

    Shows the comparison header, coverage line, aggregate metric deltas
    with direction indicators, and session transition counts.
    """
    from raki.report.html_report import METRIC_METADATA

    output_console = console or Console()
    output_console.print()

    # Header
    output_console.print(
        f"Comparing [bold]{diff.baseline_run_id}[/bold] → [bold]{diff.compare_run_id}[/bold]"
    )

    # Coverage line
    match_result = diff.match_result
    matched_count = len(match_result.matched_ids)
    total_count = max(match_result.baseline_total, match_result.compare_total)
    new_count = len(match_result.new_ids)
    dropped_count = len(match_result.dropped_ids)

    coverage_parts: list[str] = []
    if new_count > 0:
        coverage_parts.append(f"{new_count} new")
    if dropped_count > 0:
        coverage_parts.append(f"{dropped_count} dropped")
    coverage_detail = f" ({', '.join(coverage_parts)})" if coverage_parts else ""
    output_console.print(f"Matched: {matched_count}/{total_count} sessions{coverage_detail}")

    # Warning banner for dropped/new sessions
    if dropped_count > 0 or new_count > 0:
        warning_parts: list[str] = []
        if dropped_count > 0:
            warning_parts.append(f"{dropped_count} sessions dropped")
        if new_count > 0:
            warning_parts.append(f"{new_count} new")
        output_console.print(
            f"[yellow]Warning: {', '.join(warning_parts)} — "
            f"aggregate deltas based on matched sessions only[/yellow]"
        )

    output_console.print()

    # Aggregate deltas table
    if diff.deltas:
        for metric_delta in diff.deltas:
            meta = METRIC_METADATA.get(metric_delta.name, {})
            display_name = str(meta.get("display_name", metric_delta.name))
            display_format = str(meta.get("display_format", "score"))

            baseline_str = _format_delta_value(metric_delta.baseline_value, display_format)
            compare_str = _format_delta_value(metric_delta.compare_value, display_format)
            delta_str = _format_delta_change(metric_delta.delta, display_format)

            if metric_delta.direction == "improved":
                color = "green"
                indicator = "▲"
            elif metric_delta.direction == "regressed":
                color = "red"
                indicator = "▼"
            else:
                color = "white"
                indicator = "="

            output_console.print(
                f"[{color}]  {display_name:<20} {baseline_str} → {compare_str}  "
                f"({delta_str})  {indicator}[/{color}]"
            )
        output_console.print()

    # Session transition counts
    if diff.has_session_data:
        if diff.improvements:
            transition_labels = _group_transition_labels(diff.improvements)
            output_console.print(
                f"Improvements: {len(diff.improvements)} sessions ({transition_labels})"
            )
        if diff.regressions:
            transition_labels = _group_transition_labels(diff.regressions)
            output_console.print(
                f"Regressions:  {len(diff.regressions)} sessions ({transition_labels})"
            )
        if not diff.improvements and not diff.regressions:
            output_console.print("[dim]No session verdict changes[/dim]")
    else:
        output_console.print(
            "[yellow]Per-session comparison unavailable — "
            "both reports must be generated with --include-sessions[/yellow]"
        )


def _group_transition_labels(
    transitions: list[SessionTransition],
) -> str:
    """Group transitions by type and produce a summary label like 'REWORK → PASS'."""
    label_counter: Counter[str] = Counter()
    for trans in transitions:
        label = f"{trans.old_verdict.upper()} → {trans.new_verdict.upper()}"
        label_counter[label] += 1

    parts = [label for label, _count in label_counter.most_common()]
    return ", ".join(parts)
