from dataclasses import dataclass
from pathlib import Path

from raki.adapters.protocol import SessionAdapter
from raki.adapters.registry import AdapterRegistry
from raki.model import EvalDataset, EvalSample


@dataclass
class LoadError:
    path: Path
    error: str


class DatasetLoader:
    def __init__(self, registry: AdapterRegistry) -> None:
        self._registry = registry
        self.errors: list[LoadError] = []
        self.skipped: list[Path] = []

    def load_directory(self, root: Path) -> EvalDataset:
        # Resolve to an absolute path to prevent trivial path traversal.
        # Full manifest-level validation is deferred to the CLI/manifest layer (Issues #6/#8).
        root = root.resolve()
        samples: list[EvalSample] = []
        self.errors = []
        self.skipped = []
        for child in sorted(root.iterdir()):
            adapter = self._detect_adapter(child)
            if adapter is None:
                self.skipped.append(child)
                continue
            try:
                sample = adapter.load(child)
                samples.append(sample)
            except Exception as exc:
                self.errors.append(LoadError(path=child, error=str(exc)))
        return EvalDataset(samples=samples)

    def load_session(self, path: Path, adapter_name: str | None = None) -> EvalSample:
        path = path.resolve()
        if adapter_name:
            adapter = self._registry.get(adapter_name)
            if adapter is None:
                raise ValueError(f"Unknown adapter: {adapter_name}")
        else:
            adapter = self._detect_adapter(path)
            if adapter is None:
                raise ValueError(f"No adapter detected for {path}")
        return adapter.load(path)

    def _detect_adapter(self, path: Path) -> SessionAdapter | None:
        for adapter in self._registry.list_all():
            if adapter.detect(path):
                return adapter
        return None
