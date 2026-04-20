# Writing a Custom Adapter

Adapters bridge external session formats into RAKI's internal `EvalSample` model.
RAKI ships two built-in adapters (`session-schema` for multi-file directories,
`alcove` for single-file JSON transcripts). Write a custom adapter when your data
uses a different format.

## The SessionAdapter Protocol

Satisfy this protocol (`src/raki/adapters/protocol.py`):

```python
@runtime_checkable
class SessionAdapter(Protocol):
    name: str              # unique identifier, e.g. "my-format"
    description: str       # one-liner shown by `raki adapters`
    detection_hint: str    # what detect() looks for, e.g. "*.custom.json"

    def detect(self, source: Path) -> bool: ...
    def load(self, source: Path) -> EvalSample: ...
```

Because `ty` is strict, declare `name`, `description`, and `detection_hint` as
**class variables**, not instance attributes set in `__init__`.

## Detection

`detect(source)` answers: "Is this path in my format?" It must be **cheap** --
never parse the full file. Check file extensions, read only the first 4KB for
marker strings, reject symlinks early, and return `False` on any `OSError`.
The registry calls every adapter's `detect()` on every candidate path.

## Loading

`load(source)` parses the session into an `EvalSample`:

```python
class EvalSample(BaseModel):
    session: SessionMeta
    phases: list[PhaseResult]
    findings: list[ReviewFinding]
    events: list[SessionEvent]
    ground_truth: GroundTruth | None = None
```

`phases`, `findings`, and `events` are required; pass empty lists when the
source has none. Raise `ValueError` for unrecoverable problems.

## Redaction

**All adapters must call `redact_sensitive()` / `redact_dict()` before populating
`EvalSample`.** This is a security requirement (see AGENTS.md, "Security" section).

```python
from raki.adapters.redact import redact_sensitive, redact_dict
```

- `redact_sensitive(text)` -- strips tokens, API keys, JWTs, passwords from text.
- `redact_dict(data)` -- recursively redacts all strings in nested dicts/lists.

Use `redact_sensitive()` on free-text fields, `redact_dict()` on structured data.

## Registration

Register your adapter in `default_registry()` in `src/raki/adapters/__init__.py`:

```python
from raki.adapters.my_format import MyFormatAdapter


def default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    registry.register(AlcoveAdapter())
    registry.register(MyFormatAdapter())
    return registry
```

Run `raki adapters` to verify your adapter appears in the list.

For programmatic use, you can also call `default_registry()` directly:

```python
from raki.adapters import default_registry

registry = default_registry()
```

## Testing

Follow TDD: write failing tests first. Place fixtures under `tests/fixtures/`
and reuse factory fixtures from `conftest.py`.

**Edge cases:** `detect()` returns `False` for wrong formats, empty files,
symlinks, and OSErrors; `load()` handles missing optional fields; `load()` raises
`ValueError` for corrupt data; secrets never leak into `EvalSample`; `detect()`
reads at most 4KB.

## Complete Example

Minimal adapter for a CSV format (`session_id,phase_name,status,cost_usd,output`):

```python
import csv
from datetime import datetime, timezone
from pathlib import Path

from raki.adapters.redact import redact_sensitive
from raki.model import EvalSample, PhaseResult, SessionMeta

DETECT_READ_SIZE = 4096


class CsvSessionAdapter:
    name: str = "csv-session"
    description: str = "CSV with session_id, phase_name, status, cost_usd, output columns"
    detection_hint: str = "*.csv with session_id header"

    def detect(self, source: Path) -> bool:
        if source.is_symlink() or not source.is_file() or source.suffix != ".csv":
            return False
        try:
            with source.open(encoding="utf-8", errors="replace") as fh:
                header = fh.read(DETECT_READ_SIZE)
            return "session_id" in header and "phase_name" in header
        except OSError:
            return False

    def load(self, source: Path) -> EvalSample:
        with source.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"Empty CSV session file: {source}")

        session_id = rows[0]["session_id"]
        phases: list[PhaseResult] = []
        total_cost = 0.0

        for idx, row in enumerate(rows):
            cost = float(row.get("cost_usd") or 0)
            total_cost += cost
            phases.append(PhaseResult(
                name=row["phase_name"],
                generation=idx + 1,
                status=row.get("status", "completed"),
                cost_usd=cost or None,
                output=redact_sensitive(row.get("output", "")),
            ))

        meta = SessionMeta(
            session_id=session_id,
            started_at=datetime.now(timezone.utc),
            total_cost_usd=total_cost or None,
            total_phases=len(phases),
            rework_cycles=0,
        )
        return EvalSample(session=meta, phases=phases, findings=[], events=[])
```
