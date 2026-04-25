Add ``raki gate-check`` command for SODA pipeline quality evaluation.

Run ``raki gate-check <report.json>`` to evaluate a RAKI report against
SODA-tuned quality thresholds (``first_pass_success_rate>=0.6``,
``rework_cycles<=0.5``). Supports ``--baseline`` for regression detection
across milestones, ``--gate`` for custom thresholds, and ``--json`` for
CI integration. See ``docs/soda-pipeline-gate.md`` for the full workflow.
