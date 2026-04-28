# Session Schema Reference

The session-schema format represents an agentic session as a directory containing structured JSON files. Each session directory must contain `meta.json` and `events.jsonl`; phase files are optional.

## meta.json

Top-level session metadata.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket` | string | Yes | Ticket or issue identifier for the session |
| `summary` | string | No | Brief description of the ticket |
| `branch` | string | No | Git branch associated with the work |
| `started_at` | ISO 8601 datetime | Yes | When the session began |
| `total_cost` | float | No | Total cost in USD across all phases |
| `rework_cycles` | integer | No | Number of rework iterations (default 0) |
| `model_id` | string | No | LLM model used for the session |
| `phases` | object | No | Per-phase metadata (see below) |

### phases (nested in meta.json)

Each key in `phases` is a phase name (e.g., `triage`, `implement`). Values are objects with:

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | string | No | Phase outcome: `completed`, `failed`, or `skipped` |
| `cost` | float | No | Phase cost in USD |
| `duration_ms` | integer | No | Phase duration in milliseconds |
| `tokens_in` | integer | No | Input tokens consumed |
| `tokens_out` | integer | No | Output tokens generated |
| `generation` | integer | No | Generation number for the base phase file |

## triage.json

Output from the triage phase, assessing ticket complexity.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `complexity` | string | No | Estimated complexity (e.g., `low`, `medium`, `high`) |
| `approach` | string | No | Planned implementation approach |
| `automatable` | boolean | No | Whether the task can be fully automated |

## plan.json

Output from the planning phase. Like `triage.json`, this is a freeform JSON object capturing the implementation plan.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `approach` | string | No | Planned implementation approach |
| `steps` | array of strings | No | Planned implementation steps |
| `estimated_complexity` | string | No | Estimated complexity (e.g., `low`, `medium`, `high`) |

## implement.json

Output from the implementation phase. Suffixed files (e.g., `implement.json.1`, `implement.json.2`) represent earlier generations in rework cycles; the unsuffixed file is the latest generation.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `commits` | array of strings | No | Commit hashes produced |
| `files_changed` | array of strings | No | Files modified during implementation |
| `tests_passed` | boolean | No | Whether tests passed after implementation |

## verify.json

Output from the verification phase.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `verdict` | string | No | Verification result (e.g., `pass`, `fail`) |
| `command_results` | array of objects | No | Results from verification commands |
| `criteria_results` | array of objects | No | Results per acceptance criterion |

## review.json

Output from the review phase. Suffixed files (e.g., `review.json.1`) represent earlier review generations.

The adapter supports two finding formats:

**Legacy flat format** — findings at the top level:

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `verdict` | string | No | Review outcome (e.g., `approved`, `changes_requested`) |
| `findings` | array of objects | No | List of review findings (see below) |

**SODA perspectives format** — findings nested inside reviewer perspectives:

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `verdict` | string | No | Review outcome: `approve` or `rework` |
| `perspectives` | array of objects | No | Per-specialist review perspectives (see below) |

### findings (nested in review.json — flat format)

| Field | Type | Required | Description |
|---|---|---|---|
| `source` | string | No | Reviewer identifier (defaults to `unknown`) |
| `severity` | string | Yes | Finding severity: `critical`, `major`, or `minor` |
| `file` | string | No | File path related to the finding |
| `line` | integer | No | Line number related to the finding |
| `issue` | string | Yes | Description of the issue found |
| `suggestion` | string | No | Suggested fix or improvement |

### perspectives (nested in review.json — SODA format)

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Specialist name (e.g., `python`, `security`); used as reviewer |
| `verdict` | string | Yes | Specialist outcome: `clean` or `needs_fixes` |
| `findings` | array of objects | Yes | Per-finding list (same shape as flat format, but `severity` uses uppercase: `CRITICAL`, `IMPORTANT`, `MINOR`) |

Severity mapping from SODA uppercase labels to raki values: `CRITICAL` → `critical`, `IMPORTANT` → `major`, `MINOR` → `minor`.

## submit.json

Output from the submit phase. Records the pull request created from the implementation.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `pr_url` | string | Yes | URL of the opened pull request |
| `pr_number` | integer | Yes | Pull request number |
| `title` | string | Yes | Pull request title |
| `branch` | string | Yes | Source branch submitted |
| `target` | string | Yes | Target branch (e.g., `main`) |

## monitor.json

Output from the monitor phase. Records CI results and review comment handling after the PR is open.

| Field | Type | Required | Description |
|---|---|---|---|
| `ticket_key` | string | Yes | Ticket identifier |
| `pr_url` | string | Yes | URL of the monitored pull request |
| `comments_handled` | array of objects | Yes | Review comments and how they were handled |
| `tests_passed` | boolean | Yes | Whether post-merge CI checks passed |

### comments_handled (nested in monitor.json)

| Field | Type | Required | Description |
|---|---|---|---|
| `comment_id` | string | Yes | Unique identifier for the review comment |
| `author` | string | Yes | Author of the review comment |
| `content` | string | No | Text of the comment |
| `action` | string | Yes | How the comment was handled: `fixed`, `explained`, or `deferred` |
| `response` | string | Yes | Description of the action taken |

## events.jsonl

Newline-delimited JSON log of session events. Each line is a JSON object.

| Field | Type | Required | Description |
|---|---|---|---|
| `timestamp` | ISO 8601 datetime | Yes | When the event occurred |
| `phase` | string | No | Phase the event belongs to (null for session-level events) |
| `kind` | string (Literal) | Yes | Event type (see values below) |
| `data` | object | No | Event-specific payload (default `{}`) |

### kind values

| Value | Description |
|---|---|
| `phase_started` | A phase has begun execution |
| `phase_completed` | A phase finished successfully |
| `phase_failed` | A phase finished with an error |
| `review_merged` | Multiple reviewer outputs were merged |
| `review_rework_routed` | Review findings routed back for rework |
| `rework_feedback_injected` | Rework feedback was injected into the next iteration |
| `reviewer_started` | A reviewer agent started its review |
| `reviewer_completed` | A reviewer agent completed its review |
