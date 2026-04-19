"""Ground truth loading from curated YAML and domain-based matching to sessions."""

from pathlib import Path

import yaml

from raki.model import EvalSample
from raki.model.ground_truth import GroundTruth


def load_ground_truth(path: Path) -> list[GroundTruth]:
    """Load ground truth entries from a curated YAML file.

    Uses ``GroundTruth(**item)`` pattern so the Pydantic model is the single
    source of truth for field definitions.  Non-model keys (e.g. ``id``,
    ``source``) present in the YAML are silently filtered out.

    Args:
        path: Path to a YAML file containing a list of ground truth entries.

    Returns:
        A list of validated GroundTruth instances.  Returns an empty list
        when the file is empty or does not contain a YAML list.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        return []
    entries: list[GroundTruth] = []
    model_fields = set(GroundTruth.model_fields.keys())
    for item in raw:
        if not isinstance(item, dict):
            continue
        filtered = {key: value for key, value in item.items() if key in model_fields}
        entries.append(GroundTruth(**filtered))
    return entries


def match_ground_truth(
    sample: EvalSample,
    entries: list[GroundTruth],
) -> GroundTruth | None:
    """Match a session sample to the best ground truth entry by domain overlap.

    Extracts domains from the triage phase's ``output_structured["code_area"]``
    field and finds the ground truth entry with the highest domain overlap.

    Args:
        sample: The evaluation sample to match against.
        entries: Available ground truth entries.

    Returns:
        The best-matching GroundTruth entry, or None if no overlap exists.
    """
    session_domains = _extract_domains(sample)
    if not session_domains:
        return None
    best_match: GroundTruth | None = None
    best_overlap = 0
    for entry in entries:
        overlap = len(session_domains & set(entry.domains))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = entry
    return best_match


def _extract_domains(sample: EvalSample) -> set[str]:
    """Extract domain tokens from a sample's triage phase output.

    Looks at ``output_structured["code_area"]`` in any triage-phase result
    and splits the value on commas and whitespace to produce domain tokens.

    Args:
        sample: The evaluation sample to extract domains from.

    Returns:
        A set of lower-cased domain token strings.
    """
    domains: set[str] = set()
    for phase in sample.phases:
        if phase.name == "triage" and phase.output_structured:
            code_area = phase.output_structured.get("code_area", "")
            if isinstance(code_area, str):
                for token in code_area.replace(",", " ").split():
                    domains.add(token.lower().strip())
    return domains
