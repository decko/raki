Synthesize ``ReviewFinding`` objects from Alcove/bridge transcript tool failures.

When an Alcove transcript contains test or lint command failures (e.g. ``pytest``,
``ruff``, ``cargo test``) and no explicit ``findings`` array is provided in the JSON,
the adapter now generates ``ReviewFinding`` objects directly from the failure output.
Each synthesized finding has:

- ``finding_source="synthesized"`` — a new discriminant field on ``ReviewFinding``.
- ``severity="major"``, ``reviewer="synthesized"``.
- Duplicate failure texts are collapsed to one finding (e.g. repeated test runs).

Explicit findings parsed from the ``findings`` JSON key are tagged
``finding_source="review"`` so callers can always tell the two apart.

**Metric impact**

- ``review_severity_distribution`` — synthesized findings *do* count (real
  code quality signal).
- ``knowledge_gap_rate`` / ``knowledge_miss_rate`` — synthesized findings
  *excluded* from both the ``doc_chunks`` path and the legacy
  ``knowledge_context`` path (raw tool output matches too broadly).
- ``self_correction_rate`` — synthesized findings *excluded* from the
  denominator (they are not actionable reviewer feedback).

**CLI & HTML**

- Summary sentence notes synthesized findings: e.g.
  *"Reviewers found 2 major issues (3 synthesized from test failures)"*.
- HTML drill-down badges synthesized findings with a grey *"synthesized"* label
  and shows an explanatory footnote.
