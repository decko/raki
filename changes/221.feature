Add ``raki import-history`` command to backfill ``history.jsonl`` from existing session directories.

The new command discovers sessions under one or more input paths using the adapter registry (session-schema and alcove formats are both supported), computes all operational metrics without making any LLM calls, and appends one ``HistoryEntry`` per session to the JSONL history file.  Sessions already present in the history are automatically skipped so repeated imports are idempotent.

Key options:

- ``--history-path`` — override the default ``.raki/history.jsonl`` destination
- ``--adapter`` — force a specific adapter instead of auto-detecting
- ``--dry-run`` — preview what would be imported without writing anything
- ``-q / --quiet`` — suppress per-session output lines

Two new public helpers are also available in ``raki.report.history``:

- ``load_run_ids(history_path)`` — return the set of ``run_id`` values already in the history file (O(1) deduplication)
- ``import_history_entry(entry, history_path, existing_ids)`` — append a ``HistoryEntry`` when its ``run_id`` is not already present, updating the caller-owned ``existing_ids`` set in-place

A new internal module ``raki.adapters.discovery`` provides ``discover_sessions(paths, registry)`` which walks input paths recursively and returns all detected session paths, respecting symlink safety and deduplication.
