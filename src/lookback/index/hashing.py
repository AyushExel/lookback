"""Path and content hashing helpers used by the indexer.

- ``file_id_for_path`` — stable ID derived from the absolute path. Survives
  file content changes; identifies *this file* over time.
- ``content_hash_for_path`` — sha256 of the file's bytes, streamed so we
  don't blow memory on large files. The indexer compares it against the
  previously-stored hash to decide whether to re-index.
- ``chunk_id`` — stable per-chunk ID for the chunks table primary key.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_READ_CHUNK = 64 * 1024


def file_id_for_path(path: Path) -> str:
    """sha256 of the absolute path string. Stable across content edits."""
    return hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()


def content_hash_for_path(path: Path) -> str:
    """sha256 of file bytes, streamed in 64 KiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_READ_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def chunk_id(file_id: str, chunk_idx: int) -> str:
    """sha256 of ``file_id || chunk_idx``. Stable as long as ordering is stable."""
    h = hashlib.sha256()
    h.update(file_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(chunk_idx).encode("utf-8"))
    return h.hexdigest()
