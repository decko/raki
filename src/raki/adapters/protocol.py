from pathlib import Path
from typing import Protocol, runtime_checkable

from raki.model import EvalSample


@runtime_checkable
class SessionAdapter(Protocol):
    name: str

    def detect(self, source: Path) -> bool: ...
    def load(self, source: Path) -> EvalSample: ...
