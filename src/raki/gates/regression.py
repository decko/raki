"""Regression detection: compare baseline and current scores for metric regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher_is_better", "lower_is_better"]


@dataclass(frozen=True)
class RegressionResult:
    """Result of checking one metric for regression between baseline and current."""

    metric: str
    baseline: float
    current: float
    direction: Direction
    regressed: bool


def detect_regressions(
    baseline_scores: dict[str, float | None],
    current_scores: dict[str, float | None],
    metric_directions: dict[str, Direction],
    noise_margin: float = 0.02,
) -> list[RegressionResult]:
    """Detect regressions between baseline and current scores.

    For each metric present in both score dicts, check whether the score
    moved in the wrong direction by more than the noise margin.

    Args:
        baseline_scores: Metric scores from the baseline run.
        current_scores: Metric scores from the current run.
        metric_directions: Map of metric names to direction
            ("higher_is_better" or "lower_is_better").
        noise_margin: Ignore regressions smaller than this absolute
            difference (default 0.02 = 2%).

    Returns:
        List of RegressionResult instances for metrics present in both runs
        with non-None scores.
    """
    common_metrics = set(baseline_scores.keys()) & set(current_scores.keys())
    results: list[RegressionResult] = []

    for metric_name in sorted(common_metrics):
        baseline_value = baseline_scores[metric_name]
        current_value = current_scores[metric_name]

        if baseline_value is None or current_value is None:
            continue

        direction = metric_directions.get(metric_name, "higher_is_better")
        delta = current_value - baseline_value

        if direction == "higher_is_better":
            regressed = delta < -noise_margin
        else:
            regressed = delta > noise_margin

        results.append(
            RegressionResult(
                metric=metric_name,
                baseline=baseline_value,
                current=current_value,
                direction=direction,
                regressed=regressed,
            )
        )

    return results


def compute_exit_code(
    threshold_violated: bool,
    regression_detected: bool,
) -> int:
    """Compute the CLI exit code from threshold and regression results.

    Returns:
        0 = clear, 1 = threshold violation, 3 = regression, 4 = both.
    """
    if threshold_violated and regression_detected:
        return 4
    if threshold_violated:
        return 1
    if regression_detected:
        return 3
    return 0
