Add ``raki trends`` command — show metric trajectories over time from the JSONL history log.

``raki trends`` reads ``.raki/history.jsonl`` and displays a sparkline + delta table for every metric, grouped by tier (Operational → Knowledge → Analytical). Key options:

- ``--metrics NAMES`` — comma-separated filter to specific metrics (validates against known names)
- ``--since DATE`` / ``--until DATE`` — restrict to a time window (YYYY-MM-DD)
- ``--last N`` — cap to the most recent N evaluation runs
- ``--json`` — machine-readable output with full value series and delta
- ``--history-path PATH`` — point to a custom history file

Metric names from older raki versions are automatically translated (e.g. ``first_pass_verify_rate`` → ``first_pass_success_rate``). Runs that did not record a given metric are silently skipped (gap handling).
