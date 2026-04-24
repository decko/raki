``SessionMeta`` gains three optional fields — ``orchestrator``, ``provider``, and
``pipeline_phases`` — that expose pipeline/orchestrator metadata from each session.

Both adapters now populate these fields automatically:

- **session-schema**: ``orchestrator`` is inferred from the ``branch`` prefix (e.g.
  ``"soda/101"`` → ``"soda"``); ``pipeline_phases`` is the ordered list of phase
  names from the session's phases dict.
- **alcove / bridge**: ``orchestrator`` is ``"bridge"`` for bridge-format sessions
  and ``"alcove"`` for classic Claude Code transcripts; ``provider`` is taken from
  the top-level ``provider`` field when present; ``pipeline_phases`` is populated
  from the phases dict.

All three fields default to ``None``, so existing code and serialised data are
100% backward-compatible. (#175)
