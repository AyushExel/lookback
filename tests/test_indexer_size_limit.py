"""Feature under test: files larger than ``max_file_bytes`` are skipped and
counted in ``files_skipped``. We never want to embed an enormous PDF or log
into the user's local index.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def test_oversized_file_is_skipped(tmp_path: Path, tmp_store_dir: Path) -> None:
    big = tmp_path / "huge.md"
    big.write_text("# Big\n" + ("x" * 1024))
    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
        max_file_bytes=64,
    )
    stats = indexer.index_path(big)
    assert stats.files_skipped == 1
    assert stats.files_indexed == 0
    assert store.chunks_text_table().count_rows() == 0
