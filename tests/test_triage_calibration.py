"""Tests for TriageCalibrationMetric.

All tests use a local _make_triage_sample helper to avoid polluting conftest.py
with fixtures that are only useful for this single metric.
"""

from datetime import datetime, timezone

import pytest

from raki.metrics.operational.triage_calibration import (
    MEDIUM_MAX,
    SMALL_MAX,
    TriageCalibrationMetric,
)
from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset, EvalSample, PhaseResult, SessionMeta


def _make_triage_sample(
    session_id: str,
    complexity: str | None,
    cost: float | None,
) -> EvalSample:
    """Build an EvalSample with a triage phase containing the given complexity and cost."""
    meta = SessionMeta(
        session_id=session_id,
        started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        total_phases=2,
        rework_cycles=0,
        total_cost_usd=cost,
    )
    output_structured = {"complexity": complexity} if complexity is not None else None
    phases = [
        PhaseResult(
            name="triage",
            generation=1,
            status="completed",
            output="triage done",
            output_structured=output_structured,
        ),
        PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        ),
    ]
    return EvalSample(session=meta, phases=phases, findings=[], events=[])


class TestTriageCalibrationSmall:
    """Sessions predicted as 'small' complexity."""

    def test_small_within_threshold_is_calibrated(self):
        """small + cost <= SMALL_MAX → score 1.0."""
        sample = _make_triage_sample("s1", complexity="small", cost=SMALL_MAX - 1.0)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.details["calibrated_sessions"] == 1
        assert result.details["sessions_with_triage_and_cost"] == 1
        assert result.sample_scores["s1"] == 1.0

    def test_small_exactly_at_threshold_is_calibrated(self):
        """small + cost == SMALL_MAX → score 1.0 (inclusive boundary)."""
        sample = _make_triage_sample("s2", complexity="small", cost=SMALL_MAX)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.sample_scores["s2"] == 1.0

    def test_small_over_threshold_is_miscalibrated(self):
        """small + cost > SMALL_MAX → score 0.0."""
        sample = _make_triage_sample("s3", complexity="small", cost=SMALL_MAX + 0.01)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score == 0.0
        assert result.sample_scores["s3"] == 0.0


class TestTriageCalibrationMedium:
    """Sessions predicted as 'medium' complexity."""

    def test_medium_within_threshold_is_calibrated(self):
        """medium + cost <= MEDIUM_MAX → score 1.0."""
        sample = _make_triage_sample("m1", complexity="medium", cost=MEDIUM_MAX - 1.0)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.sample_scores["m1"] == 1.0

    def test_medium_over_threshold_is_miscalibrated(self):
        """medium + cost > MEDIUM_MAX → score 0.0."""
        sample = _make_triage_sample("m2", complexity="medium", cost=MEDIUM_MAX + 5.0)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score == 0.0
        assert result.sample_scores["m2"] == 0.0


class TestTriageCalibrationLarge:
    """Sessions predicted as 'large' complexity."""

    def test_large_any_cost_is_calibrated(self):
        """large sessions are always calibrated regardless of cost."""
        sample = _make_triage_sample("l1", complexity="large", cost=999.0)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.sample_scores["l1"] == 1.0


class TestTriageCalibrationMixed:
    """Mixed session datasets."""

    def test_mixed_complexity_levels(self):
        """Mean over calibrated/miscalibrated sessions across all complexity levels."""
        # small calibrated
        s_cal = _make_triage_sample("s-cal", complexity="small", cost=5.0)
        # small miscalibrated
        s_mis = _make_triage_sample("s-mis", complexity="small", cost=20.0)
        # medium calibrated
        m_cal = _make_triage_sample("m-cal", complexity="medium", cost=12.0)
        # large always calibrated
        l_cal = _make_triage_sample("l-cal", complexity="large", cost=100.0)

        dataset = EvalDataset(samples=[s_cal, s_mis, m_cal, l_cal])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())

        # 3 calibrated out of 4 total
        assert result.score == pytest.approx(3 / 4)
        assert result.details["calibrated_sessions"] == 3
        assert result.details["sessions_with_triage_and_cost"] == 4
        assert result.sample_scores["s-cal"] == 1.0
        assert result.sample_scores["s-mis"] == 0.0
        assert result.sample_scores["m-cal"] == 1.0
        assert result.sample_scores["l-cal"] == 1.0


class TestTriageCalibrationNA:
    """N/A conditions: missing triage phase, missing cost, invalid complexity."""

    def test_no_triage_phase_returns_na(self):
        """Sample without a triage phase is excluded → N/A."""
        meta = SessionMeta(
            session_id="no-triage",
            started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            total_phases=1,
            rework_cycles=0,
            total_cost_usd=10.0,
        )
        sample = EvalSample(
            session=meta,
            phases=[
                PhaseResult(
                    name="implement",
                    generation=1,
                    status="completed",
                    output="done",
                )
            ],
            findings=[],
            events=[],
        )
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["sessions_with_triage_and_cost"] == 0

    def test_missing_cost_returns_na(self):
        """Sample with triage but no cost is excluded → N/A."""
        sample = _make_triage_sample("no-cost", complexity="small", cost=None)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["sessions_with_triage_and_cost"] == 0

    def test_empty_dataset_returns_na(self):
        """Empty dataset → N/A."""
        dataset = EvalDataset(samples=[])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["calibrated_sessions"] == 0
        assert result.details["sessions_with_triage_and_cost"] == 0


class TestTriageCalibrationDetails:
    """Details dict structure."""

    def test_details_include_threshold_values(self):
        """Details must expose the threshold values used."""
        sample = _make_triage_sample("s1", complexity="small", cost=5.0)
        dataset = EvalDataset(samples=[sample])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.details["small_max"] == SMALL_MAX
        assert result.details["medium_max"] == MEDIUM_MAX

    def test_na_details_include_threshold_values(self):
        """N/A result must also expose threshold values."""
        dataset = EvalDataset(samples=[])
        result = TriageCalibrationMetric().compute(dataset, MetricConfig())
        assert result.details["small_max"] == SMALL_MAX
        assert result.details["medium_max"] == MEDIUM_MAX


class TestTriageCalibrationProperties:
    """Protocol attribute verification."""

    def test_properties(self):
        """All Protocol-required class attributes must be set correctly."""
        metric = TriageCalibrationMetric()
        assert metric.name == "triage_calibration"
        assert metric.requires_ground_truth is False
        assert metric.requires_llm is False
        assert metric.higher_is_better is True
        assert metric.display_format == "percent"
        assert metric.display_name == "Triage calibration"
        assert isinstance(metric.description, str) and len(metric.description) > 0
        assert isinstance(metric.rationale, str) and len(metric.rationale) > 50
