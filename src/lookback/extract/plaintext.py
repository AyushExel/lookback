"""Plaintext extractor — paragraph-aware chunking, TEXT modality."""

from __future__ import annotations

from pathlib import Path

from lookback.extract.base import ExtractedChunk, Extractor
from lookback.index.chunking import chunk_plaintext
from lookback.schema import Modality


class PlaintextExtractor(Extractor):
    extensions = frozenset({".txt", ".text", ".log", ".rst"})

    def extract(self, path: Path) -> list[ExtractedChunk]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [
            ExtractedChunk(
                text=c.text,
                modality=Modality.TEXT,
                source_kind="plaintext",
                meta=c.meta,
            )
            for c in chunk_plaintext(content)
        ]
