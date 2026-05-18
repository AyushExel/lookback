"""Extractor registry — pick the right extractor for a given path."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lookback.extract.base import Extractor
from lookback.extract.code import CodeExtractor
from lookback.extract.markdown import MarkdownExtractor
from lookback.extract.pdf import PDFExtractor
from lookback.extract.plaintext import PlaintextExtractor
from lookback.extract.screenshot import ScreenshotExtractor


class ExtractorRegistry:
    """First-match dispatch from path to extractor.

    Order matters: more-specific extensions (e.g. ``.md``) should win over
    broader fallbacks (e.g. plaintext on every text-ish file). We don't have
    that overlap today, but the ordering convention is fixed for future
    extractors.
    """

    def __init__(self, extractors: Sequence[Extractor]) -> None:
        self._extractors: list[Extractor] = list(extractors)

    def for_path(self, path: Path) -> Extractor | None:
        for e in self._extractors:
            if e.supports(path):
                return e
        return None

    def __iter__(self):
        return iter(self._extractors)

    def __len__(self) -> int:
        return len(self._extractors)


def default_registry() -> ExtractorRegistry:
    """The v0 default extractor stack."""
    return ExtractorRegistry(
        [
            MarkdownExtractor(),
            PDFExtractor(),
            CodeExtractor(),
            ScreenshotExtractor(),
            PlaintextExtractor(),
        ]
    )
