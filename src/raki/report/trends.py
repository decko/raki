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

DirectionLabel = Literal["improving", "declining", "stable", "insufficient_data"]

# Formats that use 0-1 bounded values (absolute dead-band threshold).
_PERCENT_FORMATS = {"percent", "score"}

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
    ``direction`` summarises the monotonicity of the last 3 values,
    respecting ``higher_is_better`` and a per-format dead-band.
    """

    metric_name: str
    display_name: str
    tier: TierLabel
    higher_is_better: bool
    display_format: str
    values: list[tuple[datetime, float]] = Field(default_factory=list)
    delta: float | None = None
    direction: DirectionLabel = "insufficient_data"

    model_config = {"arbitrary_types_allowed": True}


def _exceeds_dead_band(delta: float, display_format: str, reference: float) -> bool:
    """Return True when *delta* exceeds the dead-band threshold for *display_format*.

    For percent-format metrics (0-1 range), the threshold is 1 percentage point
    absolute (0.01).  For unbounded formats (count, currency, duration) the
    threshold is 1 % of *reference* with an epsilon fallback to avoid division
    by zero.
    """
    if display_format in _PERCENT_FORMATS:
        return abs(delta) > 0.01
    # Unbounded: 1 % relative threshold
    ref_magnitude = max(abs(reference), 1e-9)
    return abs(delta) > 0.01 * ref_magnitude


def _compute_direction(
    values: list[float],
    *,
    higher_is_better: bool,
    display_format: str,
) -> DirectionLabel:
    """Determine trend direction using monotonicity on the last 3 values.

    Algorithm:
    1. Fewer than 2 values → ``"insufficient_data"``.
    2. Take the last 3 values (or fewer if only 2 exist).
    3. Compute consecutive deltas between adjacent values.
    4. If all deltas exceed the dead-band in the *same* direction, classify as
       ``"improving"`` or ``"declining"`` (respecting ``higher_is_better``).
    5. Otherwise → ``"stable"``.
    """
    if len(values) < 2:
        return "insufficient_data"

    tail = values[-3:] if len(values) >= 3 else values[-2:]
    deltas = [tail[idx + 1] - tail[idx] for idx in range(len(tail) - 1)]

    # Check that every delta exceeds dead-band in the same direction
    all_positive = all(
        delta > 0 and _exceeds_dead_band(delta, display_format, tail[idx])
        for idx, delta in enumerate(deltas)
    )
    all_negative = all(
        delta < 0 and _exceeds_dead_band(delta, display_format, tail[idx])
        for idx, delta in enumerate(deltas)
    )

    if all_positive:
        return "improving" if higher_is_better else "declining"
    if all_negative:
        return "declining" if higher_is_better else "improving"
    return "stable"


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

    raw_floats = [val for _ts, val in raw_values]
    direction = _compute_direction(
        raw_floats,
        higher_is_better=higher,
        display_format=display_format,
    )

    return MetricTrend(
        metric_name=metric_name,
        display_name=display_name,
        tier=tier,
        higher_is_better=higher,
        display_format=display_format,
        values=raw_values,
        delta=delta,
        direction=direction,
    )


def sparkline(values: list[float | None], *, width: int = 10) -> str:
    """Render a Unicode block sparkline for *values*.

    Uses block elements ▁▂▃▄▅▆▇█ (8 levels).  When all values are equal,
    renders a flat midline (▄) at the 50 % level.  ``None`` entries represent
    absent metrics and are rendered as a space ``' '``.

    Args:
        values: Numeric values to render, in chronological order.
            ``None`` values represent absent metrics (gaps).
        width: Maximum character width.  Capped at 20.  When ``len(values)``
               exceeds *width*, the rightmost *width* values are used.

    Returns:
        A string of length ``min(len(values), width)`` composed of block chars
        and spaces (for ``None`` gaps).  Returns ``""`` when fewer than 3
        data points are present (including ``None`` entries).
    """
    if len(values) < 3:
        return ""

    width = min(width, 20)
    display_values = values[-width:] if len(values) > width else values

    # Collect non-None values for min/max computation
    numeric_values = [val for val in display_values if val is not None]

    if not numeric_values:
        # All gaps — return spaces
        return " " * len(display_values)

    blocks = "▁▂▃▄▅▆▇█"
    min_val = min(numeric_values)
    max_val = max(numeric_values)

    result: list[str] = []
    for val in display_values:
        if val is None:
            result.append(" ")
            continue
        if max_val == min_val:
            # Flat line — use mid-block
            result.append("▄")
        else:
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
    manifest_filter: str | None = None,
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
        manifest_filter: When provided, only history entries whose
            ``HistoryEntry.manifest`` field matches this string are included.

    Returns:
        List of :class:`MetricTrend` objects, one per discovered (or requested)
        metric, sorted by tier then name.
    """
    # Apply manifest filter
    filtered = entries
    if manifest_filter is not None:
        filtered = [entry for entry in filtered if entry.manifest == manifest_filter]
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


def _direction_markup(direction: DirectionLabel, higher_is_better: bool) -> str:
    """Return a Rich-markup string for the trend direction indicator.

    ``▲`` green for improving, ``▼`` red for declining, ``=`` white for stable,
    ``—`` dim for insufficient data.  The colour logic respects
    ``higher_is_better``: improving is always green, declining is always red.
    """
    if direction == "improving":
        return "[green]▲ improving[/green]"
    if direction == "declining":
        return "[red]▼ declining[/red]"
    if direction == "stable":
        return "[white]= stable[/white]"
    return "[dim]— n/a[/dim]"


def render_trends_table(trends: list[MetricTrend], console: Console | None = None) -> None:
    """Render *trends* as a Rich table to *console*.

    Columns: Metric (~25ch) | Current (~8ch) | History sparkline (max 10ch) |
    Trend direction (~12ch).

    Missing data (empty ``values``) is shown as ``N/A``.
    Trend direction uses ``▲`` green / ``▼`` red / ``=`` white / ``—`` dim.
    """
    from rich.console import Console as RichConsole
    from rich.table import Table

    output_console = console or RichConsole()

    if not trends:
        output_console.print("[dim]No trend data available.[/dim]")
        return

    table = Table(title="Metric Trends", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold", min_width=25)
    table.add_column("Current", justify="right", min_width=8)
    table.add_column("History", min_width=10)
    table.add_column("Trend", min_width=12)

    for trend in trends:
        run_count = len(trend.values)

        if run_count == 0:
            table.add_row(
                trend.display_name,
                "[dim]N/A[/dim]",
                "[dim]—[/dim]",
                _direction_markup(trend.direction, trend.higher_is_better),
            )
            continue

        raw_vals: list[float | None] = [val for _ts, val in trend.values]
        spark = sparkline(raw_vals)
        spark_display = spark if spark else "—"
        latest_str = _format_value(trend.values[-1][1], trend.display_format)

        table.add_row(
            trend.display_name,
            latest_str,
            spark_display,
            _direction_markup(trend.direction, trend.higher_is_better),
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
                "direction": trend.direction,
                "values": [{"timestamp": ts.isoformat(), "value": val} for ts, val in trend.values],
            }
        )
    return json.dumps({"trends": output}, indent=2)
