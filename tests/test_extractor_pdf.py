"""Feature under test: ``PDFExtractor`` — reads a real PDF (generated via
fpdf2 at test time) and emits TEXT-modality chunks tagged with the page
number in ``meta["page"]``.

A second test confirms graceful handling of a corrupt PDF: no exception, no
chunks returned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.extract.pdf import PDFExtractor
from lookback.schema import Modality

fpdf = pytest.importorskip("fpdf")


def _make_pdf(path: Path, pages: list[str]) -> Path:
    pdf = fpdf.FPDF()
    pdf.set_font("helvetica", size=12)
    for page_text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 10, page_text)
    pdf.output(str(path))
    return path


def test_extract_returns_one_chunk_per_page_with_page_meta(tmp_path: Path) -> None:
    pdf_path = _make_pdf(
        tmp_path / "two-pages.pdf",
        ["This is page one with some text.", "This is page two with other text."],
    )
    chunks = PDFExtractor().extract(pdf_path)
    pages = sorted({c.meta["page"] for c in chunks})
    assert pages == [1, 2]
    for c in chunks:
        assert c.modality is Modality.TEXT
        assert c.source_kind == "pdf"
        assert c.text is not None and c.text.strip()


def test_extract_on_corrupt_pdf_returns_empty_without_raising(tmp_path: Path) -> None:
    bad = tmp_path / "not-a-pdf.pdf"
    bad.write_bytes(b"this is definitely not a PDF")
    assert PDFExtractor().extract(bad) == []
