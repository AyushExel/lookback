"""Feature under test: ``MarkdownExtractor`` — reads a ``.md`` file and emits
TEXT-modality chunks with ``source_kind="markdown"`` and per-chunk section
metadata.
"""

from __future__ import annotations

from pathlib import Path

from lookback.extract.markdown import MarkdownExtractor
from lookback.schema import Modality


def test_supports_markdown_extensions() -> None:
    e = MarkdownExtractor()
    assert e.supports(Path("a.md"))
    assert e.supports(Path("a.markdown"))
    assert e.supports(Path("a.mdx"))
    assert not e.supports(Path("a.txt"))
    assert not e.supports(Path("a.py"))


def test_extract_returns_chunks_with_text_modality(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# A\nbody A\n\n# B\nbody B\n")
    chunks = MarkdownExtractor().extract(f)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.modality is Modality.TEXT
        assert c.source_kind == "markdown"
        assert c.image_path is None


def test_extract_carries_section_titles_through(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Intro\nfirst\n\n## Setup\nsecond\n")
    chunks = MarkdownExtractor().extract(f)
    sections = [c.meta["section"] for c in chunks]
    assert sections == ["Intro", "Setup"]
