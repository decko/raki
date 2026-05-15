Structured drill-down sections in the HTML report: the per-session expanded view now
wraps **Phases**, **Findings**, and **Metrics** in individually collapsible
``<details>`` blocks with counts in each summary label (e.g. "Phases (3)",
"Findings (2)"). Each phase item also shows the count of tool calls used and
a list of files modified (when ``PhaseResult.files_modified`` is populated).
