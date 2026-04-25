"""Metric health checks — detect degenerate and dead metrics after evaluation.

Two checks are performed for each metric:

dead_metric
    The metric is N/A for more than 95 % of sessions.  This usually means
    the sessions lack the data fields the metric needs (e.g. no token counts
    for ``token_efficiency``).  Severity: **error** — the metric is effectively
    not measured and results should not be trusted.

degenerate_metric
    Every session that *did* receive a score got the exact same value (zero
    variance).  This can indicate the metric is hard-wired to a constant or
    the underlying data has no discriminating signal.  Severity: **warning** —
    the score may be technically correct but carries no information.

Aggregate-only metrics (those with an empty ``sample_scores`` dict, such as
``review_severity_distribution``) are skipped entirely because they do not
produce per-session scores.
"""

from __future__ import annotations

from raki.model.report import MetricResult, MetricWarning

# Fraction of sessions that may be N/A before the dead-metric check fires.
_NA_RATE_THRESHOLD = 0.95


def run_health_checks(result: MetricResult, total_sessions: int) -> list[MetricWarning]:
    """Run health checks on a single metric result.

    Args:
        result: The computed ``MetricResult`` for one metric.
        total_sessions: The total number of sessions in the evaluation dataset.
            Used to calculate what fraction of sessions the metric was N/A for.

    Returns:
        A (possibly empty) list of :class:`MetricWarning` objects.  An empty
        list means the metric passed all checks.
    """
    # Aggregate-only metrics have no per-session scores — skip them.
    if not result.sample_scores:
        return []

    # Also skip if total_sessions is zero to avoid division by zero.
    if total_sessions == 0:
        return []

    warnings: list[MetricWarning] = []

    # --- dead metric check ---
    scored_count = len(result.sample_scores)
    na_rate = 1.0 - scored_count / total_sessions
    if na_rate > _NA_RATE_THRESHOLD:
        warnings.append(
            MetricWarning(
                metric_name=result.name,
                check="dead_metric",
                severity="error",
                message=(
                    f"Metric '{result.name}' is N/A for {na_rate:.0%} of sessions "
                    f"(threshold: >{_NA_RATE_THRESHOLD:.0%}). "
                    "Check that sessions contain the required data fields."
                ),
            )
        )

    # --- degenerate metric check ---
    unique_scores = set(result.sample_scores.values())
    if len(unique_scores) == 1:
        constant_value = next(iter(unique_scores))
        warnings.append(
            MetricWarning(
                metric_name=result.name,
                check="degenerate_metric",
                severity="warning",
                message=(
                    f"Metric '{result.name}' has a constant score of {constant_value} "
                    "across all sessions (zero variance). "
                    "The metric may not have discriminating signal in this dataset."
                ),
            )
        )

    return warnings
