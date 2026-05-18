"""Screenshot / image extractor — IMAGE modality, one chunk per image.

The extractor itself does not load image bytes — it produces an
``ExtractedChunk`` with ``image_path`` set, and the indexer hands that path
to the image embedder. OCR text, when added in a later milestone, will fill
the ``text`` column.
"""

from __future__ import annotations

from pathlib import Path

from lookback.extract.base import ExtractedChunk, Extractor
from lookback.schema import Modality


class ScreenshotExtractor(Extractor):
    extensions = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})

    def extract(self, path: Path) -> list[ExtractedChunk]:
        if not path.is_file():
            return []
        return [
            ExtractedChunk(
                text=None,
                modality=Modality.IMAGE,
                source_kind="screenshot",
                meta={"filename": path.name},
                image_path=path,
            )
        ]
