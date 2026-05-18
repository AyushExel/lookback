"""Feature under test: ``Indexer.index_path`` on a single markdown file.

End-to-end: extract → embed (mock) → write. The test asserts row counts in
the ``chunks_text`` and ``files`` tables and that the resulting stats line
up with the work done.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def _make_indexer(tmp_store_dir: Path) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    return indexer, store


def test_indexing_a_markdown_file_writes_chunks_and_file_row(
    tmp_path: Path,
    tmp_store_dir: Path,
) -> None:
    doc = tmp_path / "notes.md"
    doc.write_text("# A\nfirst body\n\n# B\nsecond body\n")

    indexer, store = _make_indexer(tmp_store_dir)
    stats = indexer.index_path(doc)

    assert stats.files_seen == 1
    assert stats.files_indexed == 1
    assert stats.files_unchanged == 0
    assert stats.errors == 0
    assert stats.chunks_written == 2

    assert store.chunks_text_table().count_rows() == 2
    assert store.files_table().count_rows() == 1


def test_indexing_a_screenshot_writes_to_image_table(
    tmp_path: Path,
    tmp_store_dir: Path,
) -> None:
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    indexer, store = _make_indexer(tmp_store_dir)
    stats = indexer.index_path(img)

    assert stats.files_indexed == 1
    assert stats.chunks_written == 1
    assert store.chunks_image_table().count_rows() == 1
    assert store.chunks_text_table().count_rows() == 0
