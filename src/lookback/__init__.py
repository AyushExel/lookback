"""Lookback — local-first multimodal semantic memory.

The public surface grows milestone by milestone. See ``DESIGN.md``.
"""

from lookback.schema import (
    ChunkRecord,
    FileRecord,
    Modality,
    chunks_image_schema,
    chunks_text_schema,
    files_schema,
)
from lookback.store.lance_store import LanceStore

__all__ = [
    "ChunkRecord",
    "FileRecord",
    "LanceStore",
    "Modality",
    "chunks_image_schema",
    "chunks_text_schema",
    "files_schema",
]

__version__ = "0.1.1"
