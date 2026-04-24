"""Tests for metric trend computation — raki.report.trends."""

from __future__ import annotations

from datetime import datetime, timezone


from conftest import make_history_entry
from raki.report.trends import (
    METRIC_RENAME_ALIASES,
    MetricTrend,
    _apply_aliases,
    _tier_for,
    compute_all_trends,
    compute_trend,
    render_trends_json,
    render_trends_table,
    sparkline,
)


# ---------------------------------------------------------------------------
# METRIC_RENAME_ALIASES
# ---------------------------------------------------------------------------


class TestMetricRenameAliases:
    def test_old_verify_rate_maps_to_success_rate(self) -> None:
        """Old first_pass_verify_rate must map to first_pass_success_rate."""
        assert METRIC_RENAME_ALIASES["first_pass_verify_rate"] == "first_pass_success_rate"

    def test_aliases_is_dict(self) -> None:
        """METRIC_RENAME_ALIASES must be a dict."""
        assert isinstance(METRIC_RENAME_ALIASES, dict)


# ---------------------------------------------------------------------------
# _apply_aliases
# ---------------------------------------------------------------------------


class TestApplyAliases:
    def test_translates_old_name(self) -> None:
        """_apply_aliases must translate old metric names to canonical names."""
        original = {"first_pass_verify_rate": 0.75}
        result = _apply_aliases(original)
        assert "first_pass_success_rate" in result
        assert result["first_pass_success_rate"] == 0.75
        assert "first_pass_verify_rate" not in result

    def test_canonical_name_unchanged(self) -> None:
        """Canonical names that are not aliases must pass through unchanged."""
        original = {"rework_cycles": 1.5}
        result = _apply_aliases(original)
        assert result == {"rework_cycles": 1.5}

    def test_canonical_wins_over_alias(self) -> None:
        """When both old and new name present, new canonical value is kept."""
        original = {
            "first_pass_verify_rate": 0.60,
            "first_pass_success_rate": 0.80,
        }
        result = _apply_aliases(original)
        assert result["first_pass_success_rate"] == 0.80

    def test_unknown_keys_unchanged(self) -> None:
        """Unknown metric names must be returned as-is."""
        original = {"some_future_metric": 0.5}
        result = _apply_aliases(original)
        assert result == {"some_future_metric": 0.5}

    def test_empty_dict(self) -> None:
        """Empty metrics dict returns empty dict."""
        assert _apply_aliases({}) == {}


# ---------------------------------------------------------------------------
# _tier_for
# ---------------------------------------------------------------------------


class TestTierFor:
    def test_operational_metric(self) -> None:
        assert _tier_for("first_pass_success_rate") == "Operational"

    def test_knowledge_metric(self) -> None:
        assert _tier_for("knowledge_gap_rate") == "Knowledge"

    def test_analytical_metric(self) -> None:
        assert _tier_for("faithfulness") == "Analytical"

    def test_unknown_metric_is_analytical(self) -> None:
        """Unknown metrics default to Analytical tier."""
        assert _tier_for("totally_unknown_metric") == "Analytical"


# ---------------------------------------------------------------------------
# make_history_entry factory (from conftest)
# ---------------------------------------------------------------------------


class TestMakeHistoryEntry:
    def test_default_values(self) -> None:
        """make_history_entry must produce a valid HistoryEntry with defaults."""
        from raki.report.history import HistoryEntry

        entry = make_history_entry()
        assert isinstance(entry, HistoryEntry)
        assert entry.run_id == "eval-001"
        assert entry.sessions_count == 10
        assert "first_pass_success_rate" in entry.metrics

    def test_custom_timestamp(self) -> None:
        ts = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        entry = make_history_entry(timestamp=ts)
        assert entry.timestamp == ts

    def test_custom_metrics(self) -> None:
        entry = make_history_entry(metrics={"rework_cycles": 2.1})
        assert entry.metrics == {"rework_cycles": 2.1}

    def test_custom_run_id(self) -> None:
        entry = make_history_entry(run_id="my-run")
        assert entry.run_id == "my-run"


# ---------------------------------------------------------------------------
# compute_trend
# ---------------------------------------------------------------------------


class TestComputeTrend:
    def test_empty_entries_returns_empty_values(self) -> None:
        """compute_trend with no entries returns a MetricTrend with empty values."""
        trend = compute_trend([], "rework_cycles")
        assert isinstance(trend, MetricTrend)
        assert trend.values == []
        assert trend.delta is None

    def test_single_entry_no_delta(self) -> None:
        """Single entry produces one value point and delta=None."""
        entry = make_history_entry(metrics={"rework_cycles": 1.5})
        trend = compute_trend([entry], "rework_cycles")
        assert len(trend.values) == 1
        assert trend.delta is None

    def test_two_entries_delta_computed(self) -> None:
        """Two entries produce delta = latest - oldest."""
        entry_a = make_history_entry(
            run_id="run-a",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            metrics={"first_pass_success_rate": 0.70},
        )
        entry_b = make_history_entry(
            run_id="run-b",
            timestamp=datetime(2026, 4, 10, tzinfo=timezone.utc),
            metrics={"first_pass_success_rate": 0.85},
        )
        trend = compute_trend([entry_a, entry_b], "first_pass_success_rate")
        assert len(trend.values) == 2
        assert abs(trend.delta - 0.15) < 1e-9

    def test_values_sorted_oldest_first(self) -> None:
        """Values must be sorted ascending by timestamp regardless of input order."""
        entry_late = make_history_entry(
            run_id="late",
            timestamp=datetime(2026, 4, 10, tzinfo=timezone.utc),
            metrics={"rework_cycles": 2.0},
        )
        entry_early = make_history_entry(
            run_id="early",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            metrics={"rework_cycles": 1.0},
        )
        trend = compute_trend([entry_late, entry_early], "rework_cycles")
        assert trend.values[0][1] == 1.0  # earliest value first
        assert trend.values[1][1] == 2.0

    def test_entries_without_metric_are_skipped(self) -> None:
        """Entries that lack the metric are treated as gaps and skipped."""
        entry_with = make_history_entry(metrics={"rework_cycles": 1.5})
        entry_without = make_history_entry(run_id="no-data", metrics={"cost_efficiency": 5.0})
        trend = compute_trend([entry_with, entry_without], "rework_cycles")
        assert len(trend.values) == 1

    def test_alias_translated_in_entries(self) -> None:
        """Old metric names in entries are translated via METRIC_RENAME_ALIASES."""
        entry = make_history_entry(metrics={"first_pass_verify_rate": 0.70})
        trend = compute_trend([entry], "first_pass_success_rate")
        assert len(trend.values) == 1
        assert trend.values[0][1] == 0.70

    def test_metric_name_preserved(self) -> None:
        """MetricTrend.metric_name must equal the requested metric name."""
        trend = compute_trend([], "cost_efficiency")
        assert trend.metric_name == "cost_efficiency"

    def test_display_name_from_metadata(self) -> None:
        """display_name must come from METRIC_METADATA."""
        trend = compute_trend([], "rework_cycles")
        assert trend.display_name == "Rework cycles"

    def test_higher_is_better_from_metadata(self) -> None:
        """higher_is_better must reflect METRIC_METADATA (rework_cycles = False)."""
        trend = compute_trend([], "rework_cycles")
        assert trend.higher_is_better is False

    def test_tier_operational(self) -> None:
        trend = compute_trend([], "first_pass_success_rate")
        assert trend.tier == "Operational"

    def test_tier_knowledge(self) -> None:
        trend = compute_trend([], "knowledge_gap_rate")
        assert trend.tier == "Knowledge"

    def test_display_format_from_metadata(self) -> None:
        trend = compute_trend([], "cost_efficiency")
        assert trend.display_format == "currency"

    def test_negative_delta_when_decreasing(self) -> None:
        """delta must be negative when the metric decreases over time."""
        entry_a = make_history_entry(
            run_id="a",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            metrics={"first_pass_success_rate": 0.90},
        )
        entry_b = make_history_entry(
            run_id="b",
            timestamp=datetime(2026, 4, 10, tzinfo=timezone.utc),
            metrics={"first_pass_success_rate": 0.75},
        )
        trend = compute_trend([entry_a, entry_b], "first_pass_success_rate")
        assert trend.delta is not None
        assert trend.delta < 0


# ---------------------------------------------------------------------------
# sparkline
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_empty_values_returns_empty_string(self) -> None:
        assert sparkline([]) == ""

    def test_single_value_flat_midline(self) -> None:
        """Single value produces a single mid-block."""
        result = sparkline([0.5])
        assert result == "▄"

    def test_all_equal_values_flat_midline(self) -> None:
        """All-equal values produce all mid-blocks (▄)."""
        result = sparkline([0.5, 0.5, 0.5])
        assert all(char == "▄" for char in result)

    def test_length_matches_values(self) -> None:
        """Output length must equal the number of input values (when ≤ width)."""
        result = sparkline([1.0, 2.0, 3.0])
        assert len(result) == 3

    def test_width_caps_output_length(self) -> None:
        """Output must not exceed the specified width."""
        result = sparkline(list(range(50)), width=10)
        assert len(result) == 10

    def test_width_capped_at_20(self) -> None:
        """Width is capped at 20 regardless of the width argument."""
        result = sparkline(list(range(100)), width=50)
        assert len(result) == 20

    def test_shows_rightmost_values_when_truncated(self) -> None:
        """When len(values) > width, the rightmost values are used."""
        # High values at end — sparkline should reflect them
        values = [0.0] * 5 + [1.0] * 5
        result = sparkline(values, width=5)
        # Should show the last 5 values (all 1.0), which are all equal → flat
        assert all(char == "▄" for char in result)

    def test_uses_block_characters(self) -> None:
        """Output must consist entirely of Unicode block characters."""
        blocks = set("▁▂▃▄▅▆▇█")
        result = sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert all(char in blocks for char in result)

    def test_increasing_values_ascending_blocks(self) -> None:
        """Strictly increasing values should produce non-decreasing block levels."""
        result = sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        blocks = "▁▂▃▄▅▆▇█"
        indices = [blocks.index(char) for char in result]
        assert indices == sorted(indices)

    def test_min_is_lowest_block(self) -> None:
        """The minimum value maps to the lowest block ▁."""
        result = sparkline([0.0, 1.0])
        assert result[0] == "▁"

    def test_max_is_highest_block(self) -> None:
        """The maximum value maps to the highest block █."""
        result = sparkline([0.0, 1.0])
        assert result[-1] == "█"


# ---------------------------------------------------------------------------
# compute_all_trends
# ---------------------------------------------------------------------------


class TestComputeAllTrends:
    def test_empty_entries_returns_empty_list(self) -> None:
        result = compute_all_trends([])
        assert result == []

    def test_discovers_all_metrics(self) -> None:
        """All metrics in entries must appear in the returned trends."""
        entries = [
            make_history_entry(
                metrics={
                    "first_pass_success_rate": 0.80,
                    "rework_cycles": 1.5,
                    "knowledge_gap_rate": 0.10,
                }
            )
        ]
        trends = compute_all_trends(entries)
        metric_names = {trend.metric_name for trend in trends}
        assert "first_pass_success_rate" in metric_names
        assert "rework_cycles" in metric_names
        assert "knowledge_gap_rate" in metric_names

    def test_tier_ordering(self) -> None:
        """Trends must be ordered Operational → Knowledge → Analytical."""
        entries = [
            make_history_entry(
                metrics={
                    "faithfulness": 0.90,
                    "knowledge_gap_rate": 0.10,
                    "first_pass_success_rate": 0.80,
                }
            )
        ]
        trends = compute_all_trends(entries)
        tiers = [trend.tier for trend in trends]
        # Find tier transitions — they must go forward, never backward
        tier_order = {"Operational": 0, "Knowledge": 1, "Analytical": 2}
        ranks = [tier_order[tier] for tier in tiers]
        assert ranks == sorted(ranks)

    def test_metric_filter_limits_results(self) -> None:
        """metric_filter must restrict results to the requested names."""
        entries = [
            make_history_entry(metrics={"first_pass_success_rate": 0.80, "rework_cycles": 1.5})
        ]
        trends = compute_all_trends(entries, metric_filter={"rework_cycles"})
        assert len(trends) == 1
        assert trends[0].metric_name == "rework_cycles"

    def test_since_filter_excludes_old_entries(self) -> None:
        """Entries before 'since' must be excluded from trend computation."""
        old = make_history_entry(
            run_id="old",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            metrics={"rework_cycles": 3.0},
        )
        recent = make_history_entry(
            run_id="recent",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            metrics={"rework_cycles": 1.0},
        )
        cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
        trends = compute_all_trends([old, recent], since=cutoff)
        rework_trend = next(
            (trend for trend in trends if trend.metric_name == "rework_cycles"), None
        )
        assert rework_trend is not None
        assert len(rework_trend.values) == 1
        assert rework_trend.values[0][1] == 1.0

    def test_until_filter_excludes_future_entries(self) -> None:
        """Entries after 'until' must be excluded."""
        early = make_history_entry(
            run_id="early",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            metrics={"rework_cycles": 1.0},
        )
        later = make_history_entry(
            run_id="later",
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            metrics={"rework_cycles": 0.5},
        )
        cutoff = datetime(2026, 4, 10, tzinfo=timezone.utc)
        trends = compute_all_trends([early, later], until=cutoff)
        rework_trend = next(
            (trend for trend in trends if trend.metric_name == "rework_cycles"), None
        )
        assert rework_trend is not None
        assert len(rework_trend.values) == 1
        assert rework_trend.values[0][1] == 1.0

    def test_alias_translated_in_all_trends(self) -> None:
        """Old metric names in entries are translated before computing all trends."""
        entries = [make_history_entry(metrics={"first_pass_verify_rate": 0.70})]
        trends = compute_all_trends(entries)
        names = {trend.metric_name for trend in trends}
        assert "first_pass_success_rate" in names
        assert "first_pass_verify_rate" not in names

    def test_returns_metric_trend_instances(self) -> None:
        """compute_all_trends must return MetricTrend objects."""
        entries = [make_history_entry(metrics={"rework_cycles": 1.0})]
        trends = compute_all_trends(entries)
        for trend in trends:
            assert isinstance(trend, MetricTrend)

    def test_metric_filter_unknown_name_empty_values(self) -> None:
        """Requesting a metric not in entries produces empty values, not an error."""
        entries = [make_history_entry(metrics={"rework_cycles": 1.0})]
        trends = compute_all_trends(entries, metric_filter={"faithfulness"})
        assert len(trends) == 1
        assert trends[0].metric_name == "faithfulness"
        assert trends[0].values == []


# ---------------------------------------------------------------------------
# render_trends_table
# ---------------------------------------------------------------------------


class TestRenderTrendsTable:
    def test_renders_without_error(self) -> None:
        """render_trends_table must complete without raising."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        render_trends_table(trends, console=con)
        output = buf.getvalue()
        assert len(output) > 0

    def test_shows_no_data_message_when_empty(self) -> None:
        """When no trends are provided, must print a 'no data' message."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        render_trends_table([], console=con)
        assert "No trend data" in buf.getvalue()

    def test_displays_metric_display_name(self) -> None:
        """Display name from METRIC_METADATA must appear in the table."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        render_trends_table(trends, console=con)
        assert "Rework cycles" in buf.getvalue()

    def test_displays_run_count(self) -> None:
        """Run count column must appear in the rendered output."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        render_trends_table(trends, console=con)
        # Run count is 1
        assert "1" in buf.getvalue()


# ---------------------------------------------------------------------------
# render_trends_json
# ---------------------------------------------------------------------------


class TestRenderTrendsJson:
    def test_returns_valid_json(self) -> None:
        """render_trends_json must return parseable JSON."""
        import json

        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        result = render_trends_json(trends)
        data = json.loads(result)
        assert "trends" in data

    def test_json_structure(self) -> None:
        """Each trend entry must have the expected fields."""
        import json

        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        data = json.loads(render_trends_json(trends))
        trend_entry = data["trends"][0]
        assert "metric_name" in trend_entry
        assert "display_name" in trend_entry
        assert "tier" in trend_entry
        assert "higher_is_better" in trend_entry
        assert "display_format" in trend_entry
        assert "run_count" in trend_entry
        assert "delta" in trend_entry
        assert "values" in trend_entry

    def test_values_have_timestamp_and_value(self) -> None:
        """Each value entry must have 'timestamp' and 'value' keys."""
        import json

        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        data = json.loads(render_trends_json(trends))
        value_entry = data["trends"][0]["values"][0]
        assert "timestamp" in value_entry
        assert "value" in value_entry

    def test_empty_trends_returns_valid_json(self) -> None:
        """Empty trends list returns valid JSON with empty trends array."""
        import json

        result = render_trends_json([])
        data = json.loads(result)
        assert data == {"trends": []}

    def test_delta_none_when_single_run(self) -> None:
        """delta must be null in JSON when only one run is present."""
        import json

        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trends = compute_all_trends(entries)
        data = json.loads(render_trends_json(trends))
        assert data["trends"][0]["delta"] is None

    def test_run_count_matches_values_length(self) -> None:
        """run_count must equal len(values)."""
        import json

        entries = [
            make_history_entry(
                run_id="a",
                timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
                metrics={"rework_cycles": 1.5},
            ),
            make_history_entry(
                run_id="b",
                timestamp=datetime(2026, 4, 10, tzinfo=timezone.utc),
                metrics={"rework_cycles": 1.2},
            ),
        ]
        trends = compute_all_trends(entries)
        data = json.loads(render_trends_json(trends))
        trend_entry = data["trends"][0]
        assert trend_entry["run_count"] == len(trend_entry["values"])
