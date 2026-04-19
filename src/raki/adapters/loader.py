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

    def load_directory(self, root: Path, *, recursive: bool = False) -> EvalDataset:
        """Load sessions from a directory.

        Args:
            root: Root directory to scan for sessions.
            recursive: When True, descend into subdirectories to find sessions.
                Default is False for backward compatibility.

        Returns:
            EvalDataset containing all successfully loaded sessions.
        """
        # Resolve to an absolute path to prevent trivial path traversal.
        # Full manifest-level validation is deferred to the CLI/manifest layer (Issues #6/#8).
        root = root.resolve()
        samples: list[EvalSample] = []
        self.errors = []
        self.skipped = []
        self._scan_directory(root, samples, recursive=recursive)
        return EvalDataset(samples=samples)

    def _scan_directory(
        self,
        directory: Path,
        samples: list[EvalSample],
        *,
        recursive: bool,
    ) -> None:
        """Scan a directory for sessions, optionally recursing into subdirectories."""
        for child in sorted(directory.iterdir()):
            adapter = self._detect_adapter(child)
            if adapter is not None:
                try:
                    sample = adapter.load(child)
                    samples.append(sample)
                except Exception as exc:
                    self.errors.append(LoadError(path=child, error=str(exc)))
            elif recursive and child.is_dir() and not child.is_symlink():
                self._scan_directory(child, samples, recursive=True)
            else:
                self.skipped.append(child)

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
