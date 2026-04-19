"""Rich CLI summary output — color-coded metrics grouped by category."""

from collections.abc import Sequence

from rich.console import Console

from raki.metrics.protocol import Metric
from raki.model.report import EvalReport

OPERATIONAL_METRICS = {
    "first_pass_verify_rate",
    "rework_cycles",
    "review_severity_distribution",
    "cost_efficiency",
    "knowledge_retrieval_miss_rate",
}

EXPERIMENTAL_METRICS = {
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
    if not higher_is_better and display_format in ("currency", "count"):
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


def format_metric_line(
    name: str,
    score: float,
    detail: str = "",
    display_format: str = "score",
    higher_is_better: bool = True,
    sample_count: int | None = None,
    display_name: str | None = None,
) -> str:
    """Format a single metric line with color, display format, and sample count."""
    color = color_for_score(score, higher_is_better, display_format)
    label = display_name or name
    if display_format == "currency":
        score_str = f"${score:.2f}"
    elif display_format == "count":
        score_str = f"{score:.1f}"
    elif display_format == "percent":
        score_str = f"{score:.2f}"
    else:
        score_str = f"{score:.2f}"
    count_str = f" (n={sample_count})" if sample_count is not None else ""
    detail_str = f"  ({detail})" if detail else ""
    return f"[{color}]  {label:<35} {score_str}{count_str}{detail_str}[/{color}]"


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

    operational = {
        name: score
        for name, score in report.aggregate_scores.items()
        if name in OPERATIONAL_METRICS
    }
    retrieval = {
        name: score
        for name, score in report.aggregate_scores.items()
        if name not in OPERATIONAL_METRICS
    }

    if operational:
        output_console.print("[bold]Operational Health[/bold]")
        for name, score in operational.items():
            output_console.print(
                format_metric_line(
                    name,
                    score,
                    detail=meta.description(name),
                    display_format=meta.display_format(name),
                    higher_is_better=meta.higher_is_better(name),
                    display_name=meta.display_name(name),
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
            output_console.print(
                format_metric_line(
                    name,
                    score,
                    detail=meta.description(name),
                    display_format=meta.display_format(name),
                    higher_is_better=meta.higher_is_better(name),
                    display_name=meta.display_name(name),
                )
                + tag
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
