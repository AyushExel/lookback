"""Extractor base class and the unified ``ExtractedChunk`` payload.

Every extractor turns a single file path into an ordered list of
``ExtractedChunk`` records. Text-mode extractors set ``text`` and leave
``image_path`` ``None``; image-mode extractors set ``image_path`` (the path
the image embedder will load) and may set ``text`` to OCR or alt text when
available. The indexer dispatches by ``modality`` to the right embedder and
storage table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lookback.schema import Modality


@dataclass(frozen=True, slots=True)
class ExtractedChunk:
    text: str | None
    modality: Modality
    source_kind: str
    meta: dict[str, Any] = field(default_factory=dict)
    image_path: Path | None = None


class Extractor(ABC):
    """Abstract extractor. Subclasses set ``extensions`` and implement ``extract``."""

    extensions: frozenset[str] = frozenset()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def extract(self, path: Path) -> list[ExtractedChunk]:
        """Return chunks for the file at ``path``. May return an empty list."""
