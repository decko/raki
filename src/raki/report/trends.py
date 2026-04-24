"""Metric trend computation — trajectories over time from the JSONL history log."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from raki.report.cli_summary import KNOWLEDGE_METRICS, OPERATIONAL_METRICS
from raki.report.diff import is_higher_is_better
from raki.report.history import HistoryEntry
from raki.report.html_report import METRIC_METADATA

if TYPE_CHECKING:
    from rich.console import Console

# Metric name aliases for history entries written by older raki versions.
# Maps old name -> current canonical name.
METRIC_RENAME_ALIASES: dict[str, str] = {
    "first_pass_verify_rate": "first_pass_success_rate",
}

TierLabel = Literal["Operational", "Knowledge", "Analytical"]


def _tier_for(metric_name: str) -> TierLabel:
    """Determine the metric tier: Operational, Knowledge, or Analytical."""
    if metric_name in OPERATIONAL_METRICS:
        return "Operational"
    if metric_name in KNOWLEDGE_METRICS:
        return "Knowledge"
    return "Analytical"


class MetricTrend(BaseModel):
    """Trend data for a single metric across multiple evaluation runs.

    ``values`` is sorted oldest-first (ascending timestamp order).
    ``delta`` is the signed difference between the most-recent and oldest value;
    ``None`` when fewer than two data points are available.
    """

    metric_name: str
    display_name: str
    tier: TierLabel
    higher_is_better: bool
    display_format: str
    values: list[tuple[datetime, float]] = Field(default_factory=list)
    delta: float | None = None

    model_config = {"arbitrary_types_allowed": True}


def _apply_aliases(metrics: dict[str, float]) -> dict[str, float]:
    """Return a copy of *metrics* with old names translated to canonical names.

    Canonical names always win: if both old and new name are present, the new
    name's value is kept unchanged.

    Two-pass approach:
    1. First pass: add all aliased (old) names under their canonical names.
    2. Second pass: overwrite with any canonical names present directly.
    """
    result: dict[str, float] = {}
    # First pass: translate alias names (old → canonical)
    for key, value in metrics.items():
        if key in METRIC_RENAME_ALIASES:
            canonical = METRIC_RENAME_ALIASES[key]
            if canonical not in result:
                result[canonical] = value
    # Second pass: canonical (non-alias) names always overwrite
    for key, value in metrics.items():
        if key not in METRIC_RENAME_ALIASES:
            result[key] = value
    return result


def compute_trend(entries: list[HistoryEntry], metric_name: str) -> MetricTrend:
    """Compute the trend for *metric_name* across *entries*.

    Entries that do not contain *metric_name* (after alias translation) are
    treated as gaps and silently skipped, so that the trend is based solely on
    runs where the metric was active.

    Args:
        entries: History entries, may be in any order (sorted by timestamp internally).
        metric_name: Canonical metric name (post-alias).

    Returns:
        A :class:`MetricTrend` with ``values`` sorted oldest-first and ``delta``
        set to ``latest_value - oldest_value`` when at least two points exist.
    """
    meta = METRIC_METADATA.get(metric_name, {})
    display_name = str(meta.get("display_name", metric_name))
    display_format = str(meta.get("display_format", "score"))
    higher = is_higher_is_better(metric_name)
    tier = _tier_for(metric_name)

    # Collect (timestamp, value) pairs — translate aliases, skip gaps
    raw_values: list[tuple[datetime, float]] = []
    for entry in entries:
        translated = _apply_aliases(entry.metrics)
        if metric_name in translated:
            raw_values.append((entry.timestamp, translated[metric_name]))

    # Sort oldest first
    raw_values.sort(key=lambda pair: pair[0])

    delta: float | None = None
    if len(raw_values) >= 2:
        delta = raw_values[-1][1] - raw_values[0][1]

    return MetricTrend(
        metric_name=metric_name,
        display_name=display_name,
        tier=tier,
        higher_is_better=higher,
        display_format=display_format,
        values=raw_values,
        delta=delta,
    )


def sparkline(values: list[float], *, width: int = 10) -> str:
    """Render a Unicode block sparkline for *values*.

    Uses block elements ▁▂▃▄▅▆▇█ (8 levels).  When all values are equal,
    renders a flat midline (▄) at the 50 % level.

    Args:
        values: Numeric values to render, in chronological order.
        width: Maximum character width.  Capped at 20.  When ``len(values)``
               exceeds *width*, the rightmost *width* values are used.

    Returns:
        A string of length ``min(len(values), width)`` composed of block chars.
        Returns ``""`` when *values* is empty.
    """
    if not values:
        return ""

    width = min(width, 20)
    display_values = values[-width:] if len(values) > width else values

    blocks = "▁▂▃▄▅▆▇█"
    min_val = min(display_values)
    max_val = max(display_values)

    if max_val == min_val:
        # Flat line — use mid-block for all values
        return "▄" * len(display_values)

    result: list[str] = []
    for val in display_values:
        normalized = (val - min_val) / (max_val - min_val)
        index = min(int(normalized * len(blocks)), len(blocks) - 1)
        result.append(blocks[index])
    return "".join(result)


def compute_all_trends(
    entries: list[HistoryEntry],
    *,
    metric_filter: set[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[MetricTrend]:
    """Compute trends for all metrics found in *entries*.

    Metrics are ordered by tier (Operational → Knowledge → Analytical) then
    alphabetically by ``metric_name`` within each tier.

    Args:
        entries: History entries to analyze.
        metric_filter: When provided, only trends for these metric names are
            returned.  Unknown names produce ``MetricTrend`` objects with an
            empty ``values`` list.
        since: Inclusive lower bound on ``HistoryEntry.timestamp``.
        until: Inclusive upper bound on ``HistoryEntry.timestamp``.

    Returns:
        List of :class:`MetricTrend` objects, one per discovered (or requested)
        metric, sorted by tier then name.
    """
    # Apply time window filter
    filtered = entries
    if since is not None:
        since_aware = since if since.tzinfo is not None else since.replace(tzinfo=None)
        filtered = [
            entry for entry in filtered if _compare_timestamps(entry.timestamp, since_aware) >= 0
        ]
    if until is not None:
        until_aware = until if until.tzinfo is not None else until.replace(tzinfo=None)
        filtered = [
            entry for entry in filtered if _compare_timestamps(entry.timestamp, until_aware) <= 0
        ]

    # Collect all known metric names (after alias translation)
    all_metric_names: set[str] = set()
    for entry in filtered:
        translated = _apply_aliases(entry.metrics)
        all_metric_names.update(translated.keys())

    # Determine which metrics to compute
    if metric_filter is not None:
        requested = metric_filter
    else:
        requested = all_metric_names

    # Compute trends for each requested metric
    trends: list[MetricTrend] = []
    for metric_name in sorted(requested):
        trend = compute_trend(filtered, metric_name)
        trends.append(trend)

    # Sort: Operational → Knowledge → Analytical, then alphabetically within tier
    tier_order: dict[str, int] = {"Operational": 0, "Knowledge": 1, "Analytical": 2}
    trends.sort(key=lambda trend: (tier_order[trend.tier], trend.metric_name))

    return trends


def _compare_timestamps(entry_ts: datetime, bound: datetime) -> int:
    """Compare entry_ts to bound, handling timezone-aware vs naive datetimes.

    Returns negative, zero, or positive like cmp().
    """
    # Normalize both to naive UTC for comparison when tzinfo differs
    entry_naive = _to_naive_utc(entry_ts)
    bound_naive = _to_naive_utc(bound)
    if entry_naive < bound_naive:
        return -1
    if entry_naive > bound_naive:
        return 1
    return 0


def _to_naive_utc(dt: datetime) -> datetime:
    """Convert a datetime to naive UTC for comparison purposes."""
    if dt.tzinfo is None:
        return dt
    # Convert to UTC and strip tzinfo
    from datetime import timezone

    utc = dt.astimezone(timezone.utc)
    return utc.replace(tzinfo=None)


def _format_value(value: float, display_format: str) -> str:
    """Format a metric value for display in the trends table."""
    if display_format == "currency":
        return f"${value:.2f}"
    if display_format == "count":
        return f"{value:.1f}"
    if display_format == "percent":
        return f"{value:.0%}"
    if display_format == "duration":
        return f"{value:.1f}s"
    return f"{value:.2f}"


def _delta_color(delta: float, higher_is_better: bool) -> str:
    """Return the Rich color for a delta value based on direction."""
    if abs(delta) < 1e-9:
        return "white"
    if higher_is_better:
        return "green" if delta > 0 else "red"
    return "green" if delta < 0 else "red"


def _delta_str(delta: float | None, display_format: str) -> str:
    """Format a signed delta for display."""
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else ""
    if display_format == "currency":
        if delta < 0:
            return f"-${abs(delta):.2f}"
        return f"{sign}${delta:.2f}"
    if display_format == "count":
        return f"{sign}{delta:.1f}"
    if display_format == "percent":
        return f"{sign}{delta:.0%}"
    if display_format == "duration":
        return f"{sign}{delta:.1f}s"
    return f"{sign}{delta:.2f}"


def render_trends_table(trends: list[MetricTrend], console: Console | None = None) -> None:
    """Render *trends* as a Rich table to *console*.

    Columns: Metric | Tier | Runs | Sparkline | Latest | Δ (first→last)

    Missing data (empty ``values``) is shown as ``N/A``.
    Delta is colored green/red based on ``higher_is_better`` and direction.
    """
    from rich.console import Console as RichConsole
    from rich.table import Table

    output_console = console or RichConsole()

    if not trends:
        output_console.print("[dim]No trend data available.[/dim]")
        return

    table = Table(title="Metric Trends", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold", min_width=20)
    table.add_column("Tier", min_width=11)
    table.add_column("Runs", justify="right", min_width=4)
    table.add_column("Trend", min_width=10)
    table.add_column("Latest", justify="right", min_width=8)
    table.add_column("Δ (first→last)", justify="right", min_width=14)

    current_tier: str | None = None
    for trend in trends:
        if trend.tier != current_tier:
            current_tier = trend.tier

        run_count = len(trend.values)

        if run_count == 0:
            table.add_row(
                trend.display_name,
                trend.tier,
                "0",
                "[dim]—[/dim]",
                "[dim]N/A[/dim]",
                "[dim]—[/dim]",
            )
            continue

        raw_vals = [val for _ts, val in trend.values]
        spark = sparkline(raw_vals)
        latest_str = _format_value(trend.values[-1][1], trend.display_format)

        delta_text = _delta_str(trend.delta, trend.display_format)
        if trend.delta is not None:
            color = _delta_color(trend.delta, trend.higher_is_better)
            delta_markup = f"[{color}]{delta_text}[/{color}]"
        else:
            delta_markup = "[dim]—[/dim]"

        table.add_row(
            trend.display_name,
            trend.tier,
            str(run_count),
            spark,
            latest_str,
            delta_markup,
        )

    output_console.print(table)


def render_trends_json(trends: list[MetricTrend]) -> str:
    """Serialize *trends* to a JSON string.

    Each trend entry includes:
    - ``metric_name``
    - ``display_name``
    - ``tier``
    - ``higher_is_better``
    - ``display_format``
    - ``run_count``
    - ``delta`` (null when < 2 runs)
    - ``values``: list of ``{"timestamp": ISO-8601, "value": float}``
    """
    output: list[dict] = []
    for trend in trends:
        output.append(
            {
                "metric_name": trend.metric_name,
                "display_name": trend.display_name,
                "tier": trend.tier,
                "higher_is_better": trend.higher_is_better,
                "display_format": trend.display_format,
                "run_count": len(trend.values),
                "delta": trend.delta,
                "values": [{"timestamp": ts.isoformat(), "value": val} for ts, val in trend.values],
            }
        )
    return json.dumps({"trends": output}, indent=2)
