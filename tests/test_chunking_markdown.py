"""Feature under test: ``chunk_markdown`` — header-aware splitting.

Verifies that markdown sources are split at ATX headers, that pre-header
preamble is captured as its own (header=None) chunk, that each emitted chunk
carries the enclosing section title in ``meta["section"]``, and that long
sections are further split to honour the token budget.
"""

from __future__ import annotations

from lookback.index.chunking import chunk_markdown


def test_each_section_becomes_its_own_chunk() -> None:
    md = (
        "# Intro\n"
        "Welcome paragraph.\n\n"
        "## Setup\n"
        "Setup paragraph.\n\n"
        "## Usage\n"
        "Usage paragraph.\n"
    )
    chunks = chunk_markdown(md, target_tokens=200)
    sections = [c.meta["section"] for c in chunks]
    assert sections == ["Intro", "Setup", "Usage"]


def test_preamble_before_first_header_is_kept_with_none_section() -> None:
    md = "Some intro words.\n\n# Heading\nBody.\n"
    chunks = chunk_markdown(md, target_tokens=200)
    assert chunks[0].meta["section"] is None
    assert "Some intro words." in chunks[0].text
    assert chunks[1].meta["section"] == "Heading"


def test_long_section_is_split_by_token_budget() -> None:
    long_body = "para.\n\n" * 200  # ~7 chars x 200 ~= 1400 chars ~= 350 tokens
    md = f"# Big\n{long_body}"
    chunks = chunk_markdown(md, target_tokens=50, overlap_tokens=8)
    assert len(chunks) > 1
    for c in chunks:
        assert c.meta["section"] == "Big"


def test_empty_section_body_is_skipped() -> None:
    md = "# Empty\n\n# HasContent\nbody.\n"
    chunks = chunk_markdown(md, target_tokens=200)
    assert [c.meta["section"] for c in chunks] == ["HasContent"]


def test_section_index_increments_per_section() -> None:
    md = "# A\nbody A\n\n# B\nbody B\n"
    chunks = chunk_markdown(md, target_tokens=200)
    indices = [c.meta["section_idx"] for c in chunks]
    assert indices == [1, 2]
