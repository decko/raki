Reports now distinguish the **agent model** (the LLM the agent used to complete
sessions) from the **judge model** (the LLM used to score retrieval quality).

- **CLI summary** — a new ``Agent:`` line appears at the top of ``raki run``
  output whenever sessions carry a ``model_id`` (e.g. ``claude-opus-4``).
- **HTML report header** — an **Agent model** field appears alongside Run,
  Timestamp, and Sessions when model IDs are present.
- **Diff reports** — ``DiffReport`` gains an ``agent_model_mismatch`` field
  (mirroring ``judge_config_mismatch``). Both CLI and HTML diff output now show
  a warning banner when the agent model set differs between the two runs being
  compared. The HTML diff also gained a **Judge configuration mismatch** banner
  (previously only shown in the CLI). (#179)
