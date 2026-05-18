"""Reaggregate per-sample metric scores into dataset-level means.

Per-session scores already exist in ``SampleResult.scores`` after
``MetricsEngine.run()`` populates them via ``_build_sample_results()``.
This utility collects those scores by metric name and computes their mean,
reproducing the dataset-level aggregate from the per-sample breakdown.

Limitations
-----------
- **review_severity_distribution**: This metric is aggregate-only — it
  computes a single weighted score over all findings in the dataset and does
  *not* populate ``sample_scores``.  Consequently it is absent from every
  ``SampleResult.scores`` list and will not appear in the output of
  ``reaggregate_scores()``.

- **self_correction_rate**: The engine computes this as
  ``resolved_findings / total_rework_findings`` — a ratio-of-sums across all
  rework sessions.  ``reaggregate_scores()`` instead computes the
  *mean of per-session scores* (each session contributes 0.0 or 1.0),
  which gives a different result when sessions have unequal finding counts.
  The two values converge only when every rework session has exactly one
  finding.  Downstream consumers should be aware of this discrepancy when
  comparing reaggregated output against engine aggregate_scores.
"""

from raki.model.report import SampleResult


def reaggregate_scores(sample_results: list[SampleResult]) -> dict[str, float | None]:
    """Reaggregate per-sample metric scores into dataset-level means.

    For each metric name that appears in any ``SampleResult.scores`` list:

    - Collect all non-``None`` scores from the samples that include that metric.
    - Return the arithmetic mean of those scores, or ``None`` when every
      score for the metric is ``None`` (or the metric appears nowhere).

    Metrics absent from a given sample are treated the same as a ``None``
    score for that sample: they are simply skipped and do not affect the mean.

    Parameters
    ----------
    sample_results:
        Sequence of per-session results as returned by
        ``MetricsEngine.run().sample_results``.

    Returns
    -------
    dict[str, float | None]
        Mapping of metric name → mean score (or ``None`` when no non-``None``
        scores were found).  Returns an empty dict when *sample_results* is
        empty.
    """
    all_metric_names: set[str] = set()
    scores_by_metric: dict[str, list[float]] = {}

    for sample_result in sample_results:
        for metric_result in sample_result.scores:
            all_metric_names.add(metric_result.name)
            if metric_result.score is not None:
                scores_by_metric.setdefault(metric_result.name, []).append(metric_result.score)

    result: dict[str, float | None] = {}
    for metric_name in all_metric_names:
        scores = scores_by_metric.get(metric_name, [])
        result[metric_name] = sum(scores) / len(scores) if scores else None

    return result
