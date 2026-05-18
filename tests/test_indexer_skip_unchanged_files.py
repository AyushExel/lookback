"""Feature under test: re-indexing a file whose content has not changed is a
no-op — no new chunks are written and the file is reported as ``unchanged``.

This is the bedrock of incremental indexing: if a user re-runs
``lookback index`` on the same root, only changed files should be touched.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def test_second_pass_reports_unchanged_and_writes_no_chunks(
    tmp_path: Path,
    tmp_store_dir: Path,
) -> None:
    doc = tmp_path / "stable.md"
    doc.write_text("# A\nbody A\n\n# B\nbody B\n")

    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )

    first = indexer.index_path(doc)
    assert first.files_indexed == 1
    rows_after_first = store.chunks_text_table().count_rows()
    assert rows_after_first == 2

    second = indexer.index_path(doc)
    assert second.files_indexed == 0
    assert second.files_unchanged == 1
    assert store.chunks_text_table().count_rows() == rows_after_first
