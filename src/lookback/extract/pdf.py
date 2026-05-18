"""PDF extractor — text-layer extraction via pypdf, TEXT modality.

OCR for image-only PDFs is Tier 2 and intentionally not handled here; pages
with no text layer are simply skipped. Each emitted chunk carries the
1-based ``page`` number in ``meta``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lookback.extract.base import ExtractedChunk, Extractor
from lookback.index.chunking import chunk_plaintext
from lookback.schema import Modality

logger = logging.getLogger(__name__)


class PDFExtractor(Extractor):
    extensions = frozenset({".pdf"})

    def extract(self, path: Path) -> list[ExtractedChunk]:
        try:
            import pypdf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pypdf is required for PDF extraction; install lookback[pdf]"
            ) from exc

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            logger.warning("failed to open PDF %s: %s", path, exc)
            return []

        out: list[ExtractedChunk] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.debug("page %d of %s: extract_text failed: %s", page_idx, path, exc)
                continue
            if not page_text.strip():
                continue
            for sub in chunk_plaintext(page_text):
                meta = dict(sub.meta)
                meta["page"] = page_idx
                out.append(
                    ExtractedChunk(
                        text=sub.text,
                        modality=Modality.TEXT,
                        source_kind="pdf",
                        meta=meta,
                    )
                )
        return out
