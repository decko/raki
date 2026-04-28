The session-schema adapter now loads all seven SODA pipeline phases (triage, plan, implement, verify, review, submit, monitor) as ``PhaseResult`` objects. Previously only four phases were loaded; submit and monitor outputs were silently ignored.

Review findings are now parsed from the SODA ``perspectives`` structure in addition to the legacy flat ``findings`` array. The perspective name (e.g. ``python``, ``security``) is used as the reviewer identifier. SODA severity labels (``CRITICAL``, ``IMPORTANT``, ``MINOR``) are mapped to raki values (``critical``, ``major``, ``minor``).

Context synthesis now includes submit-phase data (PR title, branch, PR URL) and monitor-phase data (post-merge test result, review comments resolved). The ``docs/session-schema.md`` reference now documents all seven phases.
