"""Per-modality extractors that turn a path into ``ExtractedChunk`` records."""

from lookback.extract.base import ExtractedChunk, Extractor
from lookback.extract.code import CodeExtractor
from lookback.extract.markdown import MarkdownExtractor
from lookback.extract.pdf import PDFExtractor
from lookback.extract.plaintext import PlaintextExtractor
from lookback.extract.registry import ExtractorRegistry, default_registry
from lookback.extract.screenshot import ScreenshotExtractor

__all__ = [
    "CodeExtractor",
    "ExtractedChunk",
    "Extractor",
    "ExtractorRegistry",
    "MarkdownExtractor",
    "PDFExtractor",
    "PlaintextExtractor",
    "ScreenshotExtractor",
    "default_registry",
]
