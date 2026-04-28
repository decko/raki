"""Session discovery — walk input paths and detect sessions via the adapter registry."""

from __future__ import annotations

from pathlib import Path

from raki.adapters.registry import AdapterRegistry


def discover_sessions(
    paths: list[Path],
    registry: AdapterRegistry,
    *,
    recursive: bool = True,
) -> list[Path]:
    """Walk *paths* and return every session path detected by any registered adapter.

    A session path is any file or directory for which at least one adapter's
    ``detect()`` method returns ``True``.  When a directory is detected as a
    session it is **not** recursed into (its children are part of the session,
    not separate sessions).  Symlinks are always skipped.

    Args:
        paths: Input paths to scan.  May be a mix of files and directories.
        registry: Adapter registry used for session detection.
        recursive: When ``True`` (the default), directories that are not
            themselves sessions are recursed into to find nested sessions.

    Returns:
        Detected session paths in discovery order (per-path, then
        alphabetically within each directory level).  Duplicates are removed
        while preserving first-seen order.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        # Check is_symlink() before resolve() so the symlink itself is detected,
        # not the eventual target (which resolve() would return).
        if raw_path.is_symlink():
            continue
        path = raw_path.resolve()
        if path.is_file():
            _check_file(path, registry, found, seen)
        elif path.is_dir():
            _walk_dir(path, registry, found, seen, recursive=recursive)

    return found


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_session(path: Path, registry: AdapterRegistry) -> bool:
    """Return True if *any* registered adapter detects *path* as a session."""
    return any(adapter.detect(path) for adapter in registry.list_all())


def _check_file(
    path: Path,
    registry: AdapterRegistry,
    found: list[Path],
    seen: set[Path],
) -> None:
    """Add *path* to *found* if it is detected as a session and not yet seen."""
    resolved = path.resolve()
    if resolved in seen:
        return
    if _is_session(resolved, registry):
        seen.add(resolved)
        found.append(resolved)


def _walk_dir(
    directory: Path,
    registry: AdapterRegistry,
    found: list[Path],
    seen: set[Path],
    *,
    recursive: bool,
) -> None:
    """Recursively walk *directory* and collect session paths.

    If *directory* itself is detected as a session it is added and recursion
    stops.  Otherwise all non-symlink children are visited in sorted order.
    """
    resolved = directory.resolve()
    if resolved in seen or resolved.is_symlink():
        return
    seen.add(resolved)

    # If this directory is itself a session, add it and stop recursing.
    if _is_session(resolved, registry):
        found.append(resolved)
        return

    if not recursive:
        return

    for child in sorted(resolved.iterdir()):
        if child.is_symlink():
            continue
        if child.is_file():
            _check_file(child, registry, found, seen)
        elif child.is_dir():
            _walk_dir(child, registry, found, seen, recursive=recursive)
