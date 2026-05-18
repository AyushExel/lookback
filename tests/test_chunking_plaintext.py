"""Feature under test: ``chunk_plaintext`` — paragraph-aware token-budget chunking.

We split first on blank lines (paragraph boundaries) and then accumulate
paragraphs into chunks until the token budget would be exceeded. Empty input
yields no chunks; short input yields one.
"""

from __future__ import annotations

from lookback.index.chunking import chunk_plaintext


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_plaintext("") == []
    assert chunk_plaintext("   \n\n   ") == []


def test_short_input_yields_one_chunk() -> None:
    chunks = chunk_plaintext("just one short line", target_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text == "just one short line"


def test_long_input_is_split_into_multiple_chunks() -> None:
    text = ("paragraph " * 100 + "\n\n") * 5  # well over target
    chunks = chunk_plaintext(text, target_tokens=50, overlap_tokens=8)
    assert len(chunks) >= 2


def test_chunks_preserve_paragraph_boundaries_when_they_fit() -> None:
    text = "P1 first.\n\nP2 second.\n\nP3 third."
    chunks = chunk_plaintext(text, target_tokens=200)
    assert len(chunks) == 1
    assert "P1 first." in chunks[0].text
    assert "P2 second." in chunks[0].text
    assert "P3 third." in chunks[0].text
