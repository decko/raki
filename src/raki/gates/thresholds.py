"""Per-metric quality gates: threshold parsing, evaluation, and formatting."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Regex matches: metric_name, operator (>=, <=, >, <), numeric value (including negative)
_THRESHOLD_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)(>=|<=|>|<)(-?[0-9]*\.?[0-9]+)$")

_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">": lambda actual, target: actual > target,
    "<": lambda actual, target: actual < target,
    ">=": lambda actual, target: actual >= target,
    "<=": lambda actual, target: actual <= target,
}


@dataclass(frozen=True)
class Threshold:
    """A single metric threshold: metric_name operator value."""

    metric: str
    operator: str
    value: float


@dataclass(frozen=True)
class ThresholdResult:
    """Result of evaluating a single threshold against an actual score."""

    threshold: Threshold
    actual: float | None
    passed: bool
    skipped: bool = False
    reason: str = ""


def parse_threshold(raw: str) -> Threshold:
    """Parse a threshold string like 'metric_name>0.85' into a Threshold.

    Supports operators: >, <, >=, <=.

    Args:
        raw: The raw threshold string (no whitespace allowed).

    Returns:
        A parsed Threshold instance.

    Raises:
        ValueError: If the string does not match the expected syntax.
    """
    match = _THRESHOLD_PATTERN.match(raw)
    if match is None:
        raise ValueError(
            f"Invalid threshold syntax: '{raw}'. "
            "Expected format: 'metric_name>0.85' (operators: >, <, >=, <=)"
        )
    metric_name = match.group(1)
    operator = match.group(2)
    value = float(match.group(3))
    return Threshold(metric=metric_name, operator=operator, value=value)


def evaluate_threshold(
    threshold: Threshold,
    scores: dict[str, float | None],
    required_metrics: set[str] | None = None,
) -> ThresholdResult:
    """Evaluate a single threshold against the actual scores.

    If a metric is absent from scores (not computed) or is None (N/A), behavior
    depends on whether it appears in required_metrics:
    - Required: returns a FAIL result with an explanatory reason.
    - Not required: returns a SKIP result (treated as passing).

    Args:
        threshold: The threshold to evaluate.
        scores: Map of metric names to their scores (None means N/A).
        required_metrics: Set of metric names that must not be N/A.

    Returns:
        A ThresholdResult indicating pass/fail/skip.
    """
    actual = scores.get(threshold.metric)
    metric_missing = threshold.metric not in scores

    if metric_missing or actual is None:
        required = required_metrics or set()
        if threshold.metric in required:
            if metric_missing:
                reason = (
                    f"Metric '{threshold.metric}' was not computed "
                    f"(did you forget --judge?). "
                    f"Available metrics: {', '.join(sorted(scores.keys()))}"
                )
            else:
                reason = f"Metric '{threshold.metric}' is N/A but required by --require-metric"
            return ThresholdResult(
                threshold=threshold,
                actual=None,
                passed=False,
                skipped=False,
                reason=reason,
            )
        if metric_missing:
            reason = f"Metric '{threshold.metric}' was not computed — skipping threshold check"
        else:
            reason = f"Metric '{threshold.metric}' is N/A — skipping threshold check"
        return ThresholdResult(
            threshold=threshold,
            actual=None,
            passed=True,
            skipped=True,
            reason=reason,
        )

    comparison_fn = _OPERATORS[threshold.operator]
    passed = comparison_fn(actual, threshold.value)
    return ThresholdResult(
        threshold=threshold,
        actual=actual,
        passed=passed,
    )


def evaluate_all(
    thresholds: list[Threshold],
    scores: dict[str, float | None],
    required_metrics: set[str] | None = None,
) -> list[ThresholdResult]:
    """Evaluate multiple thresholds against scores.

    Args:
        thresholds: List of thresholds to evaluate.
        scores: Map of metric names to their scores.
        required_metrics: Set of metric names that must not be N/A.

    Returns:
        List of ThresholdResult instances, one per threshold.
    """
    return [
        evaluate_threshold(threshold, scores, required_metrics=required_metrics)
        for threshold in thresholds
    ]


def format_threshold_results(results: list[ThresholdResult]) -> str:
    """Format threshold results as a human-readable string.

    Args:
        results: List of threshold evaluation results.

    Returns:
        Formatted multi-line string with PASS/FAIL/SKIP for each threshold.
    """
    lines: list[str] = []
    lines.append("Quality Gates:")
    for result in results:
        threshold = result.threshold
        threshold_str = f"{threshold.metric}{threshold.operator}{threshold.value}"

        if result.skipped:
            status = "SKIP"
            detail = result.reason
        elif result.passed:
            status = "PASS"
            actual_val = result.actual
            detail = f"actual={round(actual_val, 4)}" if actual_val is not None else ""
        else:
            status = "FAIL"
            if result.actual is not None:
                detail = f"actual={round(result.actual, 4)}"
            else:
                detail = result.reason

        lines.append(f"  [{status}] {threshold_str} ({detail})")

    return "\n".join(lines)
