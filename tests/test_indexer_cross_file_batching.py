"""Feature under test: the indexer buffers chunks across files and issues
bulk Lance writes per flush, not per file.

Before v0.1.2 the indexer wrote to Lance per file: one ``add_chunks``
call per modality per file, one ``upsert_files`` per file. A
1000-file Documents folder produced ~3000 Lance commits — far below
the perf-guide's "thousands of rows per batch" recommendation, and the
manifest/version overhead became visible in user-facing latency.

This test asserts the new contract: ``store.add_chunks`` and
``store.upsert_files`` are called *per flush*, not per file. The
total number of Lance write operations scales with
``ceil(total_chunks / flush_chunk_threshold)``, not with file count.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


class _CountingStore(LanceStore):
    """A LanceStore that records how many times each write method is called."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.add_chunks_calls: list[tuple[bool, int]] = []
        self.delete_calls: list[int] = []
        self.upsert_calls: list[int] = []
        self.optimize_calls: int = 0

    def add_chunks(self, chunks, *, image):  # type: ignore[override]
        chunks = list(chunks)
        self.add_chunks_calls.append((image, len(chunks)))
        return super().add_chunks(chunks, image=image)

    def delete_chunks_by_file_ids(self, file_ids):  # type: ignore[override]
        ids = list(file_ids)
        self.delete_calls.append(len(ids))
        return super().delete_chunks_by_file_ids(ids)

    def upsert_files(self, files):  # type: ignore[override]
        fs = list(files)
        self.upsert_calls.append(len(fs))
        return super().upsert_files(fs)

    def optimize(self, **kwargs):  # type: ignore[override]
        self.optimize_calls += 1
        return super().optimize(**kwargs)


def _make_indexer(
    store: LanceStore,
    *,
    flush_chunk_threshold: int = 2000,
    flush_file_threshold: int = 200,
    optimize_every_n_flushes: int = 10,
) -> Indexer:
    return Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
        flush_chunk_threshold=flush_chunk_threshold,
        flush_file_threshold=flush_file_threshold,
        optimize_every_n_flushes=optimize_every_n_flushes,
    )


def _populate(root: Path, n_files: int) -> None:
    for i in range(n_files):
        (root / f"doc_{i:03d}.md").write_text(f"# Title {i}\nBody for file {i}.\n")


def test_flush_chunk_threshold_groups_files_into_few_lance_writes(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _populate(tmp_path, n_files=50)
    store = _CountingStore(tmp_store_dir)
    indexer = _make_indexer(store, flush_chunk_threshold=10, flush_file_threshold=999)
    stats = indexer.index_path(tmp_path)

    assert stats.files_indexed == 50
    # 50 files x 1 chunk each = 50 chunks. flush_chunk_threshold=10 ->

    # 5 flushes triggered by the threshold + 1 final flush (no-op if
    # already empty). Each flush calls add_chunks twice (text + image),
    # so at most 12 add_chunks calls — far below the 100 we'd see if
    # each file flushed individually.
    assert len(store.add_chunks_calls) <= 12, (
        f"expected ≤ 12 add_chunks calls (text + image per flush), "
        f"got {len(store.add_chunks_calls)}"
    )
    # And one upsert_files call per flush, not per file.
    assert len(store.upsert_calls) <= 6
    assert sum(store.upsert_calls) == 50


def test_flush_file_threshold_triggers_flush_even_when_chunk_count_is_low(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _populate(tmp_path, n_files=25)
    store = _CountingStore(tmp_store_dir)
    indexer = _make_indexer(store, flush_chunk_threshold=9999, flush_file_threshold=10)
    stats = indexer.index_path(tmp_path)

    assert stats.files_indexed == 25
    # 25 files / 10 per flush = 2 mid-walk flushes + 1 final flush = 3.
    # Total upsert_calls (one per flush) should be exactly 3.
    assert len(store.upsert_calls) == 3
    assert sum(store.upsert_calls) == 25


def test_single_flush_when_corpus_fits_in_one_buffer(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _populate(tmp_path, n_files=5)
    store = _CountingStore(tmp_store_dir)
    indexer = _make_indexer(
        store, flush_chunk_threshold=9999, flush_file_threshold=9999
    )
    stats = indexer.index_path(tmp_path)

    assert stats.files_indexed == 5
    # Only the final flush should fire — exactly one upsert_files call.
    assert len(store.upsert_calls) == 1
    assert store.upsert_calls[0] == 5
