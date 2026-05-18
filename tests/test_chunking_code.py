"""Feature under test: ``chunk_code`` — fixed line-window chunking with overlap.

Verifies that a source file is split into overlapping line windows, that line
ranges are reported 1-based inclusive in ``meta``, that the language tag
flows through, and that empty input is a no-op.
"""

from __future__ import annotations

import pytest

from lookback.index.chunking import chunk_code


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_code("") == []
    assert chunk_code("\n\n\n") == []


def test_short_input_yields_single_chunk() -> None:
    src = "line1\nline2\nline3"
    chunks = chunk_code(src, target_lines=40, overlap_lines=4, language="python")
    assert len(chunks) == 1
    assert chunks[0].meta["line_start"] == 1
    assert chunks[0].meta["line_end"] == 3
    assert chunks[0].meta["language"] == "python"
    assert chunks[0].text == src


def test_long_input_produces_overlapping_windows() -> None:
    src = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_code(src, target_lines=20, overlap_lines=4)
    assert len(chunks) > 1
    # First window covers lines 1-20.
    assert chunks[0].meta["line_start"] == 1
    assert chunks[0].meta["line_end"] == 20
    # Second window starts at line 17 (20 - 4 + 1).
    assert chunks[1].meta["line_start"] == 17
    # Last window ends at the last line.
    assert chunks[-1].meta["line_end"] == 100


def test_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_code("x", target_lines=10, overlap_lines=10)
    with pytest.raises(ValueError):
        chunk_code("x", target_lines=10, overlap_lines=-1)


def test_rejects_nonpositive_target() -> None:
    with pytest.raises(ValueError):
        chunk_code("x", target_lines=0)
