"""Feature under test: ``PlaintextExtractor`` — reads a ``.txt`` (and similar)
file and emits TEXT-modality chunks with ``source_kind="plaintext"``.
"""

from __future__ import annotations

from pathlib import Path

from lookback.extract.plaintext import PlaintextExtractor
from lookback.schema import Modality


def test_supports_plaintext_extensions() -> None:
    e = PlaintextExtractor()
    for ext in [".txt", ".text", ".log", ".rst"]:
        assert e.supports(Path(f"a{ext}")), f"missing {ext}"
    assert not e.supports(Path("a.md"))
    assert not e.supports(Path("a.png"))


def test_extract_returns_text_chunks(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("just one line of text.")
    chunks = PlaintextExtractor().extract(f)
    assert len(chunks) == 1
    assert chunks[0].modality is Modality.TEXT
    assert chunks[0].source_kind == "plaintext"
    assert "just one line of text." in chunks[0].text  # type: ignore[arg-type]


def test_empty_file_yields_no_chunks(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert PlaintextExtractor().extract(f) == []
