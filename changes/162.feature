Add metric health checks to detect degenerate and dead metrics. After each evaluation run, RAKI now automatically checks every metric for two conditions:

- **dead_metric** (error): the metric is N/A for more than 95% of sessions, indicating the sessions lack required data fields.
- **degenerate_metric** (warning): the metric has a constant score across all sessions (zero variance), indicating no discriminating signal.

Warnings are shown in the CLI summary (``⚠ Metric health:`` block) and in the HTML report (Metric Health table). They are also persisted in the JSON report (``EvalReport.warnings``) and in the history log (``HistoryEntry.warning_count``).

Use ``--strict-warnings`` with ``raki run`` to exit non-zero when error-severity health issues are detected.
