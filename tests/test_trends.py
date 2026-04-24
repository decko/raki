"""Tests for metric trend computation — raki.report.trends."""

from __future__ import annotations

from datetime import datetime, timezone


from conftest import make_history_entry
from raki.report.trends import (
    METRIC_RENAME_ALIASES,
    MetricTrend,
    _apply_aliases,
    _compute_direction,
    _exceeds_dead_band,
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

    def test_fewer_than_3_returns_empty_string(self) -> None:
        """Fewer than 3 data points must return empty string."""
        assert sparkline([0.5]) == ""
        assert sparkline([0.5, 0.6]) == ""

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
        result = sparkline([0.0, 0.5, 1.0])
        assert result[0] == "▁"

    def test_max_is_highest_block(self) -> None:
        """The maximum value maps to the highest block █."""
        result = sparkline([0.0, 0.5, 1.0])
        assert result[-1] == "█"

    def test_none_values_rendered_as_space(self) -> None:
        """None values (absent metrics) must be rendered as space ' '."""
        result = sparkline([0.0, None, 1.0])
        assert result[1] == " "

    def test_none_gap_in_middle(self) -> None:
        """None gaps in the middle should produce spaces surrounded by blocks."""
        result = sparkline([0.0, None, 0.5, None, 1.0])
        assert result[1] == " "
        assert result[3] == " "
        # Non-None positions should be block characters
        blocks = set("▁▂▃▄▅▆▇█")
        assert result[0] in blocks
        assert result[2] in blocks
        assert result[4] in blocks

    def test_all_none_returns_spaces(self) -> None:
        """All-None values should return a string of spaces."""
        result = sparkline([None, None, None])
        assert result == "   "


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
        assert "direction" in trend_entry
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


# ---------------------------------------------------------------------------
# _compute_direction
# ---------------------------------------------------------------------------


class TestComputeDirection:
    def test_fewer_than_2_values_insufficient_data(self) -> None:
        """Fewer than 2 values must return 'insufficient_data'."""
        assert (
            _compute_direction([], higher_is_better=True, display_format="percent")
            == "insufficient_data"
        )
        assert (
            _compute_direction([0.5], higher_is_better=True, display_format="percent")
            == "insufficient_data"
        )

    def test_monotonic_increase_higher_is_better_improving(self) -> None:
        """Monotonically increasing values + higher_is_better → 'improving'."""
        values = [0.50, 0.65, 0.80]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "improving"

    def test_monotonic_decrease_higher_is_better_declining(self) -> None:
        """Monotonically decreasing values + higher_is_better → 'declining'."""
        values = [0.80, 0.65, 0.50]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "declining"

    def test_monotonic_decrease_lower_is_better_improving(self) -> None:
        """Monotonically decreasing values + lower_is_better → 'improving'."""
        values = [3.0, 2.0, 1.0]
        result = _compute_direction(values, higher_is_better=False, display_format="count")
        assert result == "improving"

    def test_monotonic_increase_lower_is_better_declining(self) -> None:
        """Monotonically increasing values + lower_is_better → 'declining'."""
        values = [1.0, 2.0, 3.0]
        result = _compute_direction(values, higher_is_better=False, display_format="count")
        assert result == "declining"

    def test_non_monotonic_stable(self) -> None:
        """Non-monotonic values → 'stable'."""
        values = [0.70, 0.80, 0.75]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "stable"

    def test_all_identical_values_stable(self) -> None:
        """All-identical values must be classified as 'stable'."""
        values = [0.80, 0.80, 0.80]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "stable"

    def test_dead_band_percent_within_threshold_stable(self) -> None:
        """Changes within percent dead-band (0.01) must be 'stable'."""
        # All deltas are 0.005 — within the 0.01 dead-band
        values = [0.800, 0.805, 0.810]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "stable"

    def test_dead_band_count_within_threshold_stable(self) -> None:
        """Changes within count relative dead-band (1%) must be 'stable'."""
        # Reference is 100.0, 1% = 1.0. Deltas of 0.5 are within dead-band.
        values = [100.0, 100.5, 101.0]
        result = _compute_direction(values, higher_is_better=False, display_format="count")
        assert result == "stable"

    def test_dead_band_count_exceeds_threshold(self) -> None:
        """Changes exceeding count relative dead-band (1%) must trigger direction."""
        # Reference ~100. Deltas of 5.0 (5%) exceed 1% dead-band.
        values = [100.0, 105.0, 110.0]
        result = _compute_direction(values, higher_is_better=False, display_format="count")
        assert result == "declining"

    def test_two_values_uses_single_delta(self) -> None:
        """With exactly 2 values, direction is based on a single delta."""
        values = [0.50, 0.80]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "improving"

    def test_more_than_3_values_uses_last_3(self) -> None:
        """When more than 3 values, direction uses only the last 3."""
        # First 2 are increasing, but last 3 are [0.9, 0.7, 0.5] — declining
        values = [0.50, 0.70, 0.90, 0.70, 0.50]
        result = _compute_direction(values, higher_is_better=True, display_format="percent")
        assert result == "declining"

    def test_currency_dead_band_relative(self) -> None:
        """Currency format uses relative dead-band (1%)."""
        # Reference ~10.0, 1% = 0.10. Delta 0.05 is within dead-band.
        values = [10.00, 10.05, 10.10]
        result = _compute_direction(values, higher_is_better=False, display_format="currency")
        assert result == "stable"

    def test_epsilon_fallback_for_zero_reference(self) -> None:
        """When reference is near zero, epsilon fallback prevents division by zero."""
        values = [0.0, 0.5, 1.0]
        result = _compute_direction(values, higher_is_better=True, display_format="count")
        assert result == "improving"


# ---------------------------------------------------------------------------
# _exceeds_dead_band
# ---------------------------------------------------------------------------


class TestExceedsDeadBand:
    def test_percent_format_exceeds(self) -> None:
        """Percent format: delta > 0.01 exceeds dead-band."""
        assert _exceeds_dead_band(0.02, "percent", 0.5) is True

    def test_percent_format_within(self) -> None:
        """Percent format: delta ≤ 0.01 is within dead-band."""
        assert _exceeds_dead_band(0.005, "percent", 0.5) is False

    def test_score_format_uses_percent_threshold(self) -> None:
        """Score format should use the same absolute threshold as percent."""
        assert _exceeds_dead_band(0.02, "score", 0.5) is True
        assert _exceeds_dead_band(0.005, "score", 0.5) is False

    def test_count_format_relative(self) -> None:
        """Count format: 1% relative to reference."""
        # reference=100, threshold=1.0. delta=2.0 exceeds.
        assert _exceeds_dead_band(2.0, "count", 100.0) is True
        # delta=0.5 within
        assert _exceeds_dead_band(0.5, "count", 100.0) is False

    def test_currency_format_relative(self) -> None:
        """Currency format: 1% relative to reference."""
        assert _exceeds_dead_band(0.2, "currency", 10.0) is True
        assert _exceeds_dead_band(0.05, "currency", 10.0) is False


# ---------------------------------------------------------------------------
# Table rendering with 5+ entries
# ---------------------------------------------------------------------------


class TestRenderTrendsTableMultipleEntries:
    def test_table_with_5_entries(self) -> None:
        """Table must render correctly with 5+ history entries."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={
                    "first_pass_success_rate": 0.70 + idx * 0.05,
                    "rework_cycles": 2.0 - idx * 0.3,
                    "cost_efficiency": 10.0 - idx * 1.0,
                },
            )
            for idx in range(5)
        ]
        trends = compute_all_trends(entries)
        render_trends_table(trends, console=con)
        output = buf.getvalue()
        # Table must contain display names
        assert "First-pass success rate" in output
        assert "Rework cycles" in output
        assert "Cost / session" in output

    def test_table_shows_trend_direction_indicators(self) -> None:
        """Table must show direction indicators (▲, ▼, =)."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"first_pass_success_rate": 0.70 + idx * 0.05},
            )
            for idx in range(5)
        ]
        trends = compute_all_trends(entries)
        render_trends_table(trends, console=con)
        output = buf.getvalue()
        # Should contain at least one direction indicator
        assert "▲" in output or "▼" in output or "=" in output

    def test_table_has_four_columns(self) -> None:
        """Table must have Metric, Current, History, and Trend columns."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False)
        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"rework_cycles": float(idx)},
            )
            for idx in range(5)
        ]
        trends = compute_all_trends(entries)
        render_trends_table(trends, console=con)
        output = buf.getvalue()
        assert "Metric" in output
        assert "Current" in output
        assert "History" in output
        assert "Trend" in output


# ---------------------------------------------------------------------------
# higher_is_better direction coloring
# ---------------------------------------------------------------------------


class TestHigherIsBetterDirectionColoring:
    def test_improving_uses_green_for_higher_is_better(self) -> None:
        """Improving trend must show green ▲ when higher_is_better=True."""
        from raki.report.trends import _direction_markup

        result = _direction_markup("improving", higher_is_better=True)
        assert "green" in result
        assert "▲" in result

    def test_declining_uses_red_for_higher_is_better(self) -> None:
        """Declining trend must show red ▼ when higher_is_better=True."""
        from raki.report.trends import _direction_markup

        result = _direction_markup("declining", higher_is_better=True)
        assert "red" in result
        assert "▼" in result

    def test_stable_uses_white(self) -> None:
        """Stable trend must show white = regardless of higher_is_better."""
        from raki.report.trends import _direction_markup

        result = _direction_markup("stable", higher_is_better=True)
        assert "white" in result
        assert "=" in result

    def test_insufficient_data_uses_dim(self) -> None:
        """Insufficient data must show dim —."""
        from raki.report.trends import _direction_markup

        result = _direction_markup("insufficient_data", higher_is_better=True)
        assert "dim" in result
        assert "—" in result

    def test_lower_is_better_decrease_is_improving(self) -> None:
        """For lower_is_better metrics, decreasing values = improving."""
        values = [3.0, 2.0, 1.0]
        direction = _compute_direction(values, higher_is_better=False, display_format="count")
        assert direction == "improving"

    def test_lower_is_better_increase_is_declining(self) -> None:
        """For lower_is_better metrics, increasing values = declining."""
        values = [1.0, 2.0, 3.0]
        direction = _compute_direction(values, higher_is_better=False, display_format="count")
        assert direction == "declining"


# ---------------------------------------------------------------------------
# Manifest filter
# ---------------------------------------------------------------------------


class TestManifestFilter:
    def test_manifest_filter_includes_matching(self) -> None:
        """compute_all_trends with manifest_filter includes matching entries."""
        entries = [
            make_history_entry(
                run_id="a",
                manifest="raki.yaml",
                metrics={"rework_cycles": 1.5},
            ),
            make_history_entry(
                run_id="b",
                manifest="other.yaml",
                metrics={"rework_cycles": 2.0},
            ),
        ]
        trends = compute_all_trends(entries, manifest_filter="raki.yaml")
        rework = next((trend for trend in trends if trend.metric_name == "rework_cycles"), None)
        assert rework is not None
        assert len(rework.values) == 1
        assert rework.values[0][1] == 1.5

    def test_manifest_filter_excludes_non_matching(self) -> None:
        """Entries with non-matching manifest must be excluded."""
        entries = [
            make_history_entry(
                run_id="a",
                manifest="other.yaml",
                metrics={"rework_cycles": 1.5},
            ),
        ]
        trends = compute_all_trends(entries, manifest_filter="raki.yaml")
        assert trends == []

    def test_manifest_filter_none_includes_all(self) -> None:
        """manifest_filter=None must include all entries."""
        entries = [
            make_history_entry(
                run_id="a",
                manifest="raki.yaml",
                metrics={"rework_cycles": 1.5},
            ),
            make_history_entry(
                run_id="b",
                manifest=None,
                metrics={"rework_cycles": 2.0},
            ),
        ]
        trends = compute_all_trends(entries, manifest_filter=None)
        rework = next((trend for trend in trends if trend.metric_name == "rework_cycles"), None)
        assert rework is not None
        assert len(rework.values) == 2


# ---------------------------------------------------------------------------
# Absent-metric sparkline gap
# ---------------------------------------------------------------------------


class TestAbsentMetricSparklineGap:
    def test_none_produces_space(self) -> None:
        """None values in sparkline input must produce space characters."""
        result = sparkline([0.5, None, 0.5])
        assert result[1] == " "

    def test_all_none_all_spaces(self) -> None:
        """All-None input (3+ values) must produce all spaces."""
        result = sparkline([None, None, None])
        assert all(char == " " for char in result)

    def test_mixed_none_and_numeric(self) -> None:
        """Mixed None and numeric values produce spaces and blocks respectively."""
        result = sparkline([0.0, None, 0.5, None, 1.0])
        blocks = set("▁▂▃▄▅▆▇█")
        assert result[0] in blocks
        assert result[1] == " "
        assert result[2] in blocks
        assert result[3] == " "
        assert result[4] in blocks

    def test_below_3_returns_empty(self) -> None:
        """Sparkline with fewer than 3 values (including None) returns empty string."""
        assert sparkline([None, None]) == ""
        assert sparkline([0.5, None]) == ""


# ---------------------------------------------------------------------------
# Direction field on MetricTrend
# ---------------------------------------------------------------------------


class TestMetricTrendDirection:
    def test_direction_populated_by_compute_trend(self) -> None:
        """compute_trend must populate the direction field."""
        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"first_pass_success_rate": 0.60 + idx * 0.10},
            )
            for idx in range(3)
        ]
        trend = compute_trend(entries, "first_pass_success_rate")
        assert trend.direction == "improving"

    def test_direction_insufficient_data_single_entry(self) -> None:
        """Single entry must result in 'insufficient_data' direction."""
        entries = [make_history_entry(metrics={"rework_cycles": 1.5})]
        trend = compute_trend(entries, "rework_cycles")
        assert trend.direction == "insufficient_data"

    def test_direction_stable_for_identical_values(self) -> None:
        """Identical values across runs must result in 'stable' direction."""
        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"first_pass_success_rate": 0.80},
            )
            for idx in range(3)
        ]
        trend = compute_trend(entries, "first_pass_success_rate")
        assert trend.direction == "stable"

    def test_direction_in_json_output(self) -> None:
        """Direction field must appear in JSON output."""
        import json

        entries = [
            make_history_entry(
                run_id=f"run-{idx}",
                timestamp=datetime(2026, 4, idx + 1, tzinfo=timezone.utc),
                metrics={"first_pass_success_rate": 0.60 + idx * 0.10},
            )
            for idx in range(3)
        ]
        trends = compute_all_trends(entries)
        data = json.loads(render_trends_json(trends))
        fps_trend = next(
            (
                trend
                for trend in data["trends"]
                if trend["metric_name"] == "first_pass_success_rate"
            ),
            None,
        )
        assert fps_trend is not None
        assert fps_trend["direction"] == "improving"
