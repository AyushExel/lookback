"""Feature under test: ``ScreenshotExtractor`` — turns a single image file
into one IMAGE-modality chunk with ``image_path`` pointing at the source
file (the embedder loads the bytes; the extractor does not).
"""

from __future__ import annotations

from pathlib import Path

from lookback.extract.screenshot import ScreenshotExtractor
from lookback.schema import Modality


def test_supports_image_extensions() -> None:
    e = ScreenshotExtractor()
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
        assert e.supports(Path(f"a{ext}"))
    assert not e.supports(Path("a.md"))
    assert not e.supports(Path("a.pdf"))


def test_extract_emits_one_image_chunk(tmp_path: Path) -> None:
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    chunks = ScreenshotExtractor().extract(img)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.modality is Modality.IMAGE
    assert c.source_kind == "screenshot"
    assert c.image_path == img
    assert c.text is None
    assert c.meta == {"filename": "shot.png"}


def test_missing_file_yields_no_chunks(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.png"
    assert ScreenshotExtractor().extract(missing) == []
