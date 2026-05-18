"""Feature under test: the indexer calls ``store.optimize()`` periodically
during a long index pass, plus once at the end.

Without periodic compaction a Lance store accumulates fragments and
versions on every write — read latency degrades over time and disk
usage bloats. The perf guide explicitly recommends
``table.optimize()`` after large writes or on schedule.

The new indexer triggers ``optimize()`` every
``optimize_every_n_flushes`` (default 10) flushes during the walk, and
once more after the final flush at end of ``index_path``.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


class _OptimizeCountingStore(LanceStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.optimize_calls = 0

    def optimize(self, **kwargs):  # type: ignore[override]
        self.optimize_calls += 1
        return super().optimize(**kwargs)


def _populate(root: Path, n_files: int) -> None:
    for i in range(n_files):
        (root / f"doc_{i:03d}.md").write_text(f"# Title {i}\nBody.\n")


def test_optimize_fires_every_n_flushes(tmp_path: Path, tmp_store_dir: Path) -> None:
    _populate(tmp_path, n_files=20)
    store = _OptimizeCountingStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
        flush_chunk_threshold=2,    # 20 files / 1 chunk each → ~10 flushes
        flush_file_threshold=2,
        optimize_every_n_flushes=3,
    )
    stats = indexer.index_path(tmp_path)

    assert stats.files_indexed == 20
    # With flush_file_threshold=2 and 20 files, walk-flushes ≈ 10.
    # Periodic optimize fires at flush count 3, 6, 9 → 3 mid-walk
    # optimizations, plus 1 final at end of index_path.
    assert stats.optimizations >= 3
    assert stats.optimizations == store.optimize_calls


def test_no_optimize_when_nothing_indexed(tmp_path: Path, tmp_store_dir: Path) -> None:
    # Empty directory — walking yields zero files.
    store = _OptimizeCountingStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    stats = indexer.index_path(tmp_path)
    assert stats.files_indexed == 0
    assert store.optimize_calls == 0
    assert stats.optimizations == 0
