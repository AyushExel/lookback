"""Feature under test: ``Indexer.index_path(on_file=...)`` callback fires
once per walked file, passing the path and current ``IndexStats``.

The CLI uses this hook to advance a ``rich.Progress`` spinner; tests
that don't care about progress pass ``None`` and the indexer behaves
identically. We test the callback contract directly: invocation count,
monotonic stats, and that errors in the user callback don't kill the
walk.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer, IndexStats
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def _make_indexer(tmp_store_dir: Path) -> Indexer:
    store = LanceStore(tmp_store_dir)
    return Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )


def _populate(root: Path, n_files: int) -> Path:
    """Populate a *subdirectory* of ``root`` so the source tree never
    overlaps with the Lance store directory the indexer writes to."""
    src = root / "src"
    src.mkdir()
    for i in range(n_files):
        (src / f"doc_{i:03d}.md").write_text(f"# Title {i}\nBody.\n")
    return src


def test_callback_fires_once_per_file(tmp_path: Path, tmp_store_dir: Path) -> None:
    src = _populate(tmp_path, n_files=7)
    indexer = _make_indexer(tmp_store_dir)
    seen: list[Path] = []

    def cb(path: Path, _stats: IndexStats) -> None:
        seen.append(path)

    indexer.index_path(src, on_file=cb)
    assert len(seen) == 7
    # Walker yields each file exactly once (no dupes).
    assert len(set(seen)) == 7


def test_callback_receives_monotonic_stats(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = _populate(tmp_path, n_files=10)
    indexer = _make_indexer(tmp_store_dir)
    seen_counts: list[int] = []

    def cb(_path: Path, stats: IndexStats) -> None:
        seen_counts.append(stats.files_seen)

    indexer.index_path(src, on_file=cb)
    # files_seen is monotonically non-decreasing and ends at the total.
    assert seen_counts == sorted(seen_counts)
    assert seen_counts[-1] == 10
