"""Diff computation — compare two evaluation runs by session matching and metric deltas."""

from dataclasses import dataclass, field
from typing import Literal

from raki.model.report import EvalReport, SampleResult
from raki.report.html_report import METRIC_METADATA, determine_verdict


@dataclass(frozen=True)
class MatchResult:
    """Result of matching sessions between baseline and compare reports."""

    matched_ids: set[str] = field(default_factory=set)
    new_ids: set[str] = field(default_factory=set)
    dropped_ids: set[str] = field(default_factory=set)
    baseline_total: int = 0
    compare_total: int = 0


@dataclass(frozen=True)
class MetricDelta:
    """Delta for a single metric between baseline and compare."""

    name: str
    baseline_value: float
    compare_value: float
    delta: float
    direction: Literal["improved", "regressed", "flat"]


@dataclass(frozen=True)
class SessionTransition:
    """A session whose verdict changed between baseline and compare."""

    session_id: str
    old_verdict: Literal["pass", "rework", "fail"]
    new_verdict: Literal["pass", "rework", "fail"]
    transition_type: Literal["improvement", "regression"]


@dataclass(frozen=True)
class DiffReport:
    """Complete diff between two evaluation runs."""

    baseline_run_id: str
    compare_run_id: str
    match_result: MatchResult
    deltas: list[MetricDelta] = field(default_factory=list)
    improvements: list[SessionTransition] = field(default_factory=list)
    regressions: list[SessionTransition] = field(default_factory=list)
    has_session_data: bool = True
    judge_config_mismatch: list[str] = field(default_factory=list)


# Verdict rank for sorting: lower rank = worse verdict
_VERDICT_RANK: dict[str, int] = {"fail": 0, "rework": 1, "pass": 2}


def is_higher_is_better(metric_name: str) -> bool:
    """Determine if a metric is higher_is_better from METRIC_METADATA.

    Defaults to True for unknown metrics.
    """
    meta = METRIC_METADATA.get(metric_name)
    if meta is None:
        return True
    return bool(meta.get("higher_is_better", True))


def compare_judge_configs(baseline: EvalReport, compare: EvalReport) -> list[str]:
    """Return warning messages when judge configs differ between two reports.

    Compares the ``llm_provider`` and ``llm_model`` fields stored in each report's
    ``config`` dict.  Returns an empty list when:

    - Neither report used a judge (``skip_llm=True`` or config absent).
    - Both reports used identical judge settings.

    Returns one warning per differing field when both used a judge with different
    settings, or a single warning when only one report used a judge.
    """
    baseline_skip = bool(baseline.config.get("skip_llm", True))
    compare_skip = bool(compare.config.get("skip_llm", True))

    # Neither used a judge — nothing to compare
    if baseline_skip and compare_skip:
        return []

    # One used a judge, the other did not
    if baseline_skip != compare_skip:
        return ["unknown baseline — cannot compare judge calibration"]

    # Both used a judge — compare individual config fields
    warnings: list[str] = []
    judge_fields = [
        ("llm_provider", "judge provider"),
        ("llm_model", "judge model"),
    ]
    for config_key, label in judge_fields:
        baseline_val = baseline.config.get(config_key)
        compare_val = compare.config.get(config_key)
        if baseline_val != compare_val:
            warnings.append(
                f"{label} differs: {baseline_val!r} (baseline) vs {compare_val!r} (compare)"
            )
    return warnings


def match_sessions(baseline: EvalReport, compare: EvalReport) -> MatchResult:
    """Match sessions between baseline and compare reports by session_id.

    Partitions into matched (in both), new (compare only), and dropped (baseline only).
    """
    baseline_ids = {
        sample_result.sample.session.session_id for sample_result in baseline.sample_results
    }
    compare_ids = {
        sample_result.sample.session.session_id for sample_result in compare.sample_results
    }

    matched = baseline_ids & compare_ids
    new = compare_ids - baseline_ids
    dropped = baseline_ids - compare_ids

    return MatchResult(
        matched_ids=matched,
        new_ids=new,
        dropped_ids=dropped,
        baseline_total=len(baseline_ids),
        compare_total=len(compare_ids),
    )


def compute_deltas(
    baseline_scores: dict[str, float | None],
    compare_scores: dict[str, float | None],
) -> list[MetricDelta]:
    """Compute deltas for metrics present in both baseline and compare.

    Direction is determined by higher_is_better from METRIC_METADATA:
    - higher_is_better=True: positive delta = improved, negative = regressed
    - higher_is_better=False: negative delta = improved, positive = regressed

    Metrics where either score is None (N/A) are skipped.
    """
    deltas: list[MetricDelta] = []
    # Only compute deltas for metrics present in both reports
    common_metrics = set(baseline_scores.keys()) & set(compare_scores.keys())

    for metric_name in common_metrics:
        baseline_value = baseline_scores[metric_name]
        compare_value = compare_scores[metric_name]
        if baseline_value is None or compare_value is None:
            continue
        delta = compare_value - baseline_value
        higher_is_better = is_higher_is_better(metric_name)

        if abs(delta) < 1e-9:
            direction: Literal["improved", "regressed", "flat"] = "flat"
        elif higher_is_better:
            direction = "improved" if delta > 0 else "regressed"
        else:
            direction = "improved" if delta < 0 else "regressed"

        deltas.append(
            MetricDelta(
                name=metric_name,
                baseline_value=baseline_value,
                compare_value=compare_value,
                delta=delta,
                direction=direction,
            )
        )

    return deltas


def compute_transitions(
    baseline_results: list[SampleResult],
    compare_results: list[SampleResult],
    matched_ids: set[str],
) -> list[SessionTransition]:
    """Compute verdict transitions for matched sessions.

    Returns transitions sorted with regressions first, then improvements.
    """
    baseline_by_id = {
        sample_result.sample.session.session_id: sample_result for sample_result in baseline_results
    }
    compare_by_id = {
        sample_result.sample.session.session_id: sample_result for sample_result in compare_results
    }

    transitions: list[SessionTransition] = []

    for session_id in matched_ids:
        baseline_sample = baseline_by_id.get(session_id)
        compare_sample = compare_by_id.get(session_id)
        if baseline_sample is None or compare_sample is None:
            continue

        old_verdict = determine_verdict(baseline_sample.sample)
        new_verdict = determine_verdict(compare_sample.sample)

        if old_verdict == new_verdict:
            continue

        old_rank = _VERDICT_RANK[old_verdict]
        new_rank = _VERDICT_RANK[new_verdict]

        transition_type: Literal["improvement", "regression"] = (
            "improvement" if new_rank > old_rank else "regression"
        )

        transitions.append(
            SessionTransition(
                session_id=session_id,
                old_verdict=old_verdict,
                new_verdict=new_verdict,
                transition_type=transition_type,
            )
        )

    # Sort: regressions first, then improvements
    transitions.sort(key=lambda trans: 0 if trans.transition_type == "regression" else 1)
    return transitions


def generate_diff_report(baseline: EvalReport, compare: EvalReport) -> DiffReport:
    """Generate a complete diff report from two evaluation reports.

    Computes session matching, metric deltas, verdict transitions, and warns
    when the judge configuration differs between the two runs.
    """
    match_result = match_sessions(baseline, compare)
    deltas = compute_deltas(baseline.aggregate_scores, compare.aggregate_scores)
    judge_warnings = compare_judge_configs(baseline, compare)

    has_session_data = bool(baseline.sample_results and compare.sample_results)

    if has_session_data:
        transitions = compute_transitions(
            baseline.sample_results,
            compare.sample_results,
            match_result.matched_ids,
        )
    else:
        transitions = []

    improvements = [trans for trans in transitions if trans.transition_type == "improvement"]
    regressions = [trans for trans in transitions if trans.transition_type == "regression"]

    return DiffReport(
        baseline_run_id=baseline.run_id,
        compare_run_id=compare.run_id,
        match_result=match_result,
        deltas=deltas,
        improvements=improvements,
        regressions=regressions,
        has_session_data=has_session_data,
        judge_config_mismatch=judge_warnings,
    )
