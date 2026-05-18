"""Feature under test: ``ExtractorRegistry`` first-match dispatch from a path
to the right ``Extractor`` instance.

The ``default_registry`` factory wires up the v0 extractor stack; this test
exercises only the dispatch behaviour, not the extraction itself.
"""

from __future__ import annotations

from pathlib import Path

from lookback.extract.code import CodeExtractor
from lookback.extract.markdown import MarkdownExtractor
from lookback.extract.pdf import PDFExtractor
from lookback.extract.plaintext import PlaintextExtractor
from lookback.extract.registry import default_registry
from lookback.extract.screenshot import ScreenshotExtractor


def test_md_dispatches_to_markdown_extractor() -> None:
    e = default_registry().for_path(Path("doc.md"))
    assert isinstance(e, MarkdownExtractor)


def test_pdf_dispatches_to_pdf_extractor() -> None:
    e = default_registry().for_path(Path("paper.pdf"))
    assert isinstance(e, PDFExtractor)


def test_py_dispatches_to_code_extractor() -> None:
    e = default_registry().for_path(Path("script.py"))
    assert isinstance(e, CodeExtractor)


def test_png_dispatches_to_screenshot_extractor() -> None:
    e = default_registry().for_path(Path("Screenshot 2026-01-01.png"))
    assert isinstance(e, ScreenshotExtractor)


def test_txt_dispatches_to_plaintext_extractor() -> None:
    e = default_registry().for_path(Path("notes.txt"))
    assert isinstance(e, PlaintextExtractor)


def test_unknown_extension_returns_none() -> None:
    assert default_registry().for_path(Path("binary.unknown")) is None
