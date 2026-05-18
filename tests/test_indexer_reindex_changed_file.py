"""Feature under test: when a file's content changes between index passes,
the indexer deletes the previous chunks for that ``file_id`` and writes the
new ones — old data must not linger.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def test_changed_file_replaces_its_chunks(
    tmp_path: Path,
    tmp_store_dir: Path,
) -> None:
    doc = tmp_path / "evolving.md"
    doc.write_text("# Only One Section\nsmall body\n")

    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )

    indexer.index_path(doc)
    assert store.chunks_text_table().count_rows() == 1

    doc.write_text("# Section A\nfirst\n\n# Section B\nsecond\n\n# Section C\nthird\n")
    stats = indexer.index_path(doc)

    assert stats.files_indexed == 1
    assert stats.files_unchanged == 0
    assert store.chunks_text_table().count_rows() == 3, (
        "should be 3 new chunks (old 1 deleted, 3 new added)"
    )
    sections = {
        row["meta"]
        for row in store.chunks_text_table()
        .search()
        .select(["meta"])
        .limit(100)
        .to_list()
    }
    # All three new sections appear in stored meta
    assert any("Section A" in (m or "") for m in sections)
    assert any("Section C" in (m or "") for m in sections)
