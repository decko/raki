"""Documentation file chunker for RAKI.

Loads project documentation, splits files into chunks based on format-aware
heading detection, and extracts domain information from directory structure.

Path safety: rejects symlinks, enforces per-file and total size limits,
guards against path traversal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_EXTENSIONS: list[str] = [".md", ".txt"]
DEFAULT_MAX_FILE_SIZE: int = 1 * 1024 * 1024  # 1MB
DEFAULT_MAX_TOTAL_SIZE: int = 50 * 1024 * 1024  # 50MB

# RST heading underline characters
_RST_HEADING_CHARS = frozenset("=-~^\"'+`:#.*_!")

# Markdown heading pattern: lines starting with 1-6 # followed by space
_MD_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+", re.MULTILINE)


@dataclass
class DocChunk:
    """A chunk of documentation text with provenance metadata."""

    text: str
    source_file: str
    domain: str


def _extract_domain(file_path: Path, docs_root: Path) -> str:
    """Extract domain from file path relative to docs root.

    The first subdirectory under docs_root becomes the domain.
    Files directly in docs_root get domain "general".

    Examples:
        docs/auth/setup.md -> "auth"
        docs/readme.md -> "general"
        docs/api/v2/endpoints/users.md -> "api"
    """
    try:
        relative = file_path.resolve().relative_to(docs_root.resolve())
    except ValueError:
        return "general"

    parts = relative.parts
    if len(parts) <= 1:
        # File is directly in docs_root
        return "general"
    return parts[0]


def _chunk_markdown(content: str) -> list[str]:
    """Split markdown content on heading patterns (# , ## , ### ).

    Each heading starts a new chunk that includes the heading text.
    If no headings are found, returns the entire content as a single chunk.
    """
    heading_positions = [match.start() for match in _MD_HEADING_PATTERN.finditer(content)]

    if not heading_positions:
        stripped = content.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []

    # If there's content before the first heading, include it as a chunk
    if heading_positions[0] > 0:
        preamble = content[: heading_positions[0]].strip()
        if preamble:
            chunks.append(preamble)

    for chunk_idx, start_pos in enumerate(heading_positions):
        if chunk_idx + 1 < len(heading_positions):
            end_pos = heading_positions[chunk_idx + 1]
        else:
            end_pos = len(content)
        chunk_text = content[start_pos:end_pos].strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _chunk_rst(content: str) -> list[str]:
    """Split RST content on underline headings (=, -, ~, etc.).

    RST headings are detected as: a text line followed by a line consisting
    entirely of one of the RST heading characters, at least as long as the
    text line.

    If no headings are found, returns the entire content as a single chunk.
    """
    lines = content.split("\n")
    heading_line_indices: list[int] = []

    for line_idx in range(1, len(lines)):
        current_line = lines[line_idx].rstrip()
        previous_line = lines[line_idx - 1].rstrip()

        if not current_line or not previous_line:
            continue

        # Check if current line is all one RST heading char and long enough
        if (
            len(current_line) >= len(previous_line)
            and len(set(current_line)) == 1
            and current_line[0] in _RST_HEADING_CHARS
        ):
            heading_line_indices.append(line_idx - 1)

    if not heading_line_indices:
        stripped = content.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []

    # Content before first heading
    if heading_line_indices[0] > 0:
        preamble = "\n".join(lines[: heading_line_indices[0]]).strip()
        if preamble:
            chunks.append(preamble)

    for chunk_idx, heading_start in enumerate(heading_line_indices):
        if chunk_idx + 1 < len(heading_line_indices):
            end_line = heading_line_indices[chunk_idx + 1]
        else:
            end_line = len(lines)
        chunk_text = "\n".join(lines[heading_start:end_line]).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _chunk_plaintext(content: str, max_chunk_size: int = 2000) -> list[str]:
    """Split plain text on double newlines (paragraph boundaries).

    Small paragraphs are merged up to max_chunk_size characters per chunk.
    """
    raw_paragraphs = re.split(r"\n\n+", content)
    paragraphs = [para.strip() for para in raw_paragraphs if para.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_chunk_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if current_chunk_length > 0 and current_chunk_length + paragraph_length > max_chunk_size:
            # Flush current chunk
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = [paragraph]
            current_chunk_length = paragraph_length
        else:
            current_chunk_parts.append(paragraph)
            current_chunk_length += paragraph_length

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def chunk_file(path: Path, docs_root: Path) -> list[DocChunk]:
    """Chunk a single file based on its format extension.

    Args:
        path: Path to the file to chunk.
        docs_root: Root directory of the documentation tree.

    Returns:
        List of DocChunk objects with text, source_file, and domain.
    """
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    suffix = path.suffix.lower()
    if suffix == ".md":
        raw_chunks = _chunk_markdown(content)
    elif suffix == ".rst":
        raw_chunks = _chunk_rst(content)
    else:
        raw_chunks = _chunk_plaintext(content)

    domain = _extract_domain(path, docs_root)
    source_file = str(path.resolve().relative_to(docs_root.resolve()))

    return [
        DocChunk(text=chunk_text, source_file=source_file, domain=domain)
        for chunk_text in raw_chunks
    ]


def _has_symlink_ancestor(file_path: Path, root: Path) -> bool:
    """Check if any parent directory between file and root is a symlink."""
    current = file_path.parent
    while current != root and current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def load_docs(
    docs_path: Path,
    extensions: list[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    max_total_size: int = DEFAULT_MAX_TOTAL_SIZE,
) -> list[DocChunk]:
    """Walk a documentation directory and chunk all matching files.

    Applies path safety checks: rejects symlinks (both files and ancestor
    directories), files outside docs_root, files exceeding per-file size
    limit, and stops loading when total size limit is reached.

    Args:
        docs_path: Root directory containing documentation files.
        extensions: File extensions to include (default: [".md", ".txt"]).
        max_file_size: Maximum size in bytes for a single file (default: 1MB).
        max_total_size: Maximum total bytes to load (default: 50MB).

    Returns:
        List of DocChunk objects from all loaded files.
    """
    if extensions is None:
        extensions = list(DEFAULT_EXTENSIONS)

    docs_root = docs_path.resolve()
    all_chunks: list[DocChunk] = []
    total_bytes_loaded = 0

    for file_path in sorted(docs_root.rglob("*")):
        if not file_path.is_file():
            continue

        # Extension filter
        if file_path.suffix.lower() not in extensions:
            continue

        # Symlink rejection (file itself)
        if file_path.is_symlink():
            logger.warning("Skipping symlink: %s", file_path)
            continue

        # Symlink rejection (ancestor directories)
        if _has_symlink_ancestor(file_path, docs_root):
            logger.warning("Skipping file under symlinked directory: %s", file_path)
            continue

        # Path traversal guard: resolved path must be under docs_root
        try:
            file_path.resolve().relative_to(docs_root)
        except ValueError:
            logger.warning("Skipping file outside docs root: %s", file_path)
            continue

        # Per-file size limit
        file_size = file_path.stat().st_size
        if file_size > max_file_size:
            logger.warning(
                "Skipping file exceeding size limit (%d > %d): %s",
                file_size,
                max_file_size,
                file_path,
            )
            continue

        # Total size limit
        if total_bytes_loaded + file_size > max_total_size:
            logger.warning(
                "Total size limit reached (%d bytes). Stopping doc loading.",
                max_total_size,
            )
            break

        total_bytes_loaded += file_size
        file_chunks = chunk_file(file_path, docs_root)
        all_chunks.extend(file_chunks)

    return all_chunks
