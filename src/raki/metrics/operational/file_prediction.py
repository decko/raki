"""File prediction accuracy metric.

Measures how well the agent's triage file prediction (``triage.output_structured["files"]``)
matches the set of files actually changed during implementation.

Score = mean per-session F1, where per-session F1 is the harmonic mean of precision and
recall computed from the predicted and actual file sets.  Returns score=None (N/A) when no
sessions have file predictions in their triage phase.

Three helpers handle normalisation and extraction:

- ``_normalize_path``: strips leading ``./`` and lowercases to make comparisons
  format-agnostic.
- ``_extract_predicted_files``: reads ``triage.output_structured["files"]`` (list of
  strings).
- ``_extract_actual_files``: reads ``implement.output_structured["files_changed"]``
  (SODA format — list of dicts with a ``path`` key), falling back to
  ``phase.files_modified`` (Alcove format — list of strings).

Both metric score and details follow the N/A display convention:
``sessions_with_file_predictions: 0`` signals the report renderer to show "N/A"
rather than a misleading 0.0.
"""

from raki.metrics.protocol import MetricConfig
from raki.model import EvalDataset, EvalSample
from raki.model.report import MetricResult


def _normalize_path(path: str) -> str:
    """Normalise a file path for comparison.

    Strips a leading ``./`` prefix (common in SODA outputs) and converts to
    lowercase so that paths that differ only in case or relative prefix are
    treated as equal.
    """
    if path.startswith("./"):
        path = path[2:]
    return path.lower()


def _extract_predicted_files(sample: EvalSample) -> set[str]:
    """Extract predicted files from the triage phase.

    Reads ``triage.output_structured["files"]`` (list of strings).  Returns an
    empty set when the triage phase is absent or the ``files`` key is missing.
    """
    for phase in sample.phases:
        if phase.name == "triage":
            output = phase.output_structured
            if not isinstance(output, dict):
                return set()
            raw_files = output.get("files")
            if not isinstance(raw_files, list):
                return set()
            return {_normalize_path(str(filepath)) for filepath in raw_files if filepath}
    return set()


def _extract_actual_files(sample: EvalSample) -> set[str]:
    """Extract actually-changed files from the implement phase.

    Tries two sources in order:
    1. ``implement.output_structured["files_changed"]`` — SODA format, list of dicts
       each with a ``"path"`` key.
    2. ``implement.files_modified`` — Alcove format, list of strings directly on the
       ``PhaseResult``.

    Returns an empty set when the implement phase is absent or neither source
    yields file data.
    """
    for phase in sample.phases:
        if phase.name == "implement":
            # SODA format: output_structured["files_changed"] is a list of dicts
            if isinstance(phase.output_structured, dict):
                files_changed = phase.output_structured.get("files_changed")
                if isinstance(files_changed, list) and files_changed:
                    result: set[str] = set()
                    for entry in files_changed:
                        if isinstance(entry, dict):
                            raw_path = entry.get("path")
                            if raw_path and isinstance(raw_path, str):
                                result.add(_normalize_path(raw_path))
                        elif isinstance(entry, str):
                            result.add(_normalize_path(entry))
                    return result
            # Alcove format: phase.files_modified is a list of strings
            if phase.files_modified:
                return {_normalize_path(filepath) for filepath in phase.files_modified}
    return set()


def _compute_f1(predicted: set[str], actual: set[str]) -> float:
    """Compute per-session F1 from predicted and actual file sets.

    Returns 0.0 when either set is empty (precision and/or recall is undefined).
    """
    if not predicted or not actual:
        return 0.0
    intersection = len(predicted & actual)
    precision = intersection / len(predicted)
    recall = intersection / len(actual)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class FilePredictionAccuracyMetric:
    """Mean per-session F1 for triage file predictions vs. actual files changed.

    Score = mean(per-session F1) over sessions with non-empty triage file
    predictions.  Each session counts equally regardless of the number of
    files it mentions.  Higher is better: 1.0 means every session's triage
    file list perfectly matched what was actually changed; 0.0 means every
    prediction was wrong.

    Returns score=None (N/A) when no sessions have triage file predictions.
    """

    name: str = "file_prediction_accuracy"
    requires_ground_truth: bool = False
    requires_llm: bool = False
    higher_is_better: bool = True
    display_format: str = "percent"
    display_name: str = "File prediction accuracy"
    description: str = "Mean F1 between triage predicted files and actual files changed"
    rationale: str = (
        "File prediction accuracy measures how closely the agent's triage-phase file "
        "list matches the files actually modified during implementation. A high score "
        "means the agent consistently identifies the right files at the start of a session, "
        "which improves planning reliability, cost forecasting, and review targeting. "
        "A low score suggests the agent's initial scope assessment is poor — either "
        "missing files that will change (low recall) or predicting files that turn out to "
        "be untouched (low precision). Per-session F1 treats both errors symmetrically. "
        "Only sessions with non-empty triage file predictions are scored; sessions without "
        "a triage phase or without a 'files' key are excluded (N/A). "
        "Target: >= 0.70 mean F1."
    )

    def compute(self, dataset: EvalDataset, config: MetricConfig) -> MetricResult:  # noqa: ARG002
        sample_scores: dict[str, float] = {}
        session_f1_values: list[float] = []

        # Micro-average accumulators
        total_true_positives = 0
        total_predicted = 0
        total_actual = 0

        for sample in dataset.samples:
            predicted_files = _extract_predicted_files(sample)
            if not predicted_files:
                # No triage file predictions → session does not qualify
                continue

            actual_files = _extract_actual_files(sample)
            session_f1 = _compute_f1(predicted_files, actual_files)
            session_id = sample.session.session_id
            sample_scores[session_id] = session_f1
            session_f1_values.append(session_f1)

            # Accumulate micro-average components
            true_positives = len(predicted_files & actual_files)
            total_true_positives += true_positives
            total_predicted += len(predicted_files)
            total_actual += len(actual_files)

        qualifying_count = len(session_f1_values)

        if qualifying_count == 0:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "sessions_with_file_predictions": 0,
                    "micro_precision": None,
                    "micro_recall": None,
                    "micro_f1": None,
                },
            )

        # Micro-average precision/recall/F1
        micro_precision: float | None
        micro_recall: float | None
        micro_f1: float | None
        if total_predicted > 0:
            micro_precision = total_true_positives / total_predicted
        else:
            micro_precision = None
        if total_actual > 0:
            micro_recall = total_true_positives / total_actual
        else:
            micro_recall = None
        if micro_precision is not None and micro_recall is not None:
            denom = micro_precision + micro_recall
            micro_f1 = 2.0 * micro_precision * micro_recall / denom if denom > 0.0 else 0.0
        else:
            micro_f1 = None

        mean_f1 = sum(session_f1_values) / qualifying_count
        return MetricResult(
            name=self.name,
            score=mean_f1,
            details={
                "sessions_with_file_predictions": qualifying_count,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            },
            sample_scores=sample_scores,
        )
