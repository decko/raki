"""Helpers for re-rendering reports from saved JSON data.

These helpers are used by the ``report`` CLI command to reconstruct metric-like
objects from serialised report data, enabling ``print_summary`` to display
results without requiring the original Metric instances.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raki.metrics.protocol import MetricConfig
    from raki.model import EvalDataset
    from raki.model.report import EvalReport, MetricResult


@dataclass(frozen=True)
class MetricStub:
    """Lightweight stand-in for ``Metric`` when re-rendering from JSON reports.

    Satisfies the full ``Metric`` protocol so that ``print_summary`` can accept
    these stubs without type errors.
    """

    name: str
    display_name: str
    description: str
    display_format: str
    higher_is_better: bool
    requires_ground_truth: bool = False
    requires_llm: bool = False

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:  # noqa: ARG002
        """Stub -- metric stubs are display-only and never compute scores."""
        raise NotImplementedError("MetricStub is display-only and cannot compute scores")


def is_session_data_stripped(report: EvalReport) -> bool:
    """Check whether session data in sample_results has been stripped.

    Returns True if any phase output is the ``<stripped>`` sentinel, indicating
    the report was generated without ``--include-sessions``.
    """
    for sample_result in report.sample_results:
        for phase in sample_result.sample.phases:
            if phase.output == "<stripped>":
                return True
    return False


def metric_stubs_from_metadata(
    aggregate_scores: dict[str, float | None],
) -> Sequence[MetricStub]:
    """Build lightweight metric-like objects from METRIC_METADATA for print_summary.

    When re-rendering from JSON, we don't have the original Metric instances.
    This function creates simple objects that satisfy the display_name,
    description, display_format, and higher_is_better attributes that
    ``_MetricMeta`` in ``cli_summary`` looks up.
    """
    from raki.report.html_report import METRIC_METADATA

    stubs: list[MetricStub] = []
    for metric_name in aggregate_scores:
        meta = METRIC_METADATA.get(
            metric_name,
            {
                "display_name": metric_name,
                "description": "",
                "display_format": "score",
                "higher_is_better": True,
            },
        )
        stubs.append(
            MetricStub(
                name=metric_name,
                display_name=str(meta["display_name"]),
                description=str(meta["description"]),
                display_format=str(meta["display_format"]),
                higher_is_better=bool(meta["higher_is_better"]),
            )
        )
    return stubs
