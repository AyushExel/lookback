"""Feature under test: ``Watcher._handle_changes`` — the pure event-batch
processor underneath the watch loop.

We drive it with synthetic ``(Change, path)`` events instead of actual
filesystem watches so the tests are deterministic and fast. The asserts
focus on the contract the long-running watcher relies on:
- modified files get re-indexed
- deleted files have their chunks and ``files`` row removed
- a path that's both modified-and-deleted in the same batch is treated
  as deleted when it no longer exists on disk
"""

from __future__ import annotations

from pathlib import Path

from watchfiles import Change

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.index.watcher import Watcher
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


def test_modified_file_gets_indexed(tmp_path: Path, tmp_store_dir: Path) -> None:
    doc = tmp_path / "notes.md"
    doc.write_text("# Title\nFresh content.\n")
    indexer, store = _make_indexer(tmp_store_dir)
    watcher = Watcher(indexer, [tmp_path])

    batch = watcher._handle_changes({(Change.modified, str(doc))})
    assert batch.files_indexed == 1
    assert batch.chunks_written >= 1
    assert store.chunks_text_table().count_rows() >= 1


def test_added_file_gets_indexed(tmp_path: Path, tmp_store_dir: Path) -> None:
    doc = tmp_path / "new.md"
    doc.write_text("# New\nBody.\n")
    indexer, store = _make_indexer(tmp_store_dir)
    watcher = Watcher(indexer, [tmp_path])

    batch = watcher._handle_changes({(Change.added, str(doc))})
    assert batch.files_indexed == 1
    assert store.files_table().count_rows() == 1


def test_deleted_file_is_removed_from_store(tmp_path: Path, tmp_store_dir: Path) -> None:
    doc = tmp_path / "doomed.md"
    doc.write_text("# Doomed\nWill be deleted.\n")
    indexer, store = _make_indexer(tmp_store_dir)
    indexer.index_path(doc)
    assert store.files_table().count_rows() == 1

    # Now simulate deletion (file is unlinked, then event delivered).
    doc.unlink()
    watcher = Watcher(indexer, [tmp_path])
    batch = watcher._handle_changes({(Change.deleted, str(doc))})

    assert batch.files_deleted == 1
    assert batch.chunks_deleted >= 1
    assert store.files_table().count_rows() == 0
    assert store.chunks_text_table().count_rows() == 0


def test_modify_then_delete_in_same_batch_is_treated_as_delete(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    doc = tmp_path / "ephemeral.md"
    doc.write_text("# Ephemeral\nbody\n")
    indexer, store = _make_indexer(tmp_store_dir)
    indexer.index_path(doc)

    doc.unlink()  # file is gone before we process the events
    watcher = Watcher(indexer, [tmp_path])
    batch = watcher._handle_changes(
        {(Change.modified, str(doc)), (Change.deleted, str(doc))}
    )
    assert batch.files_indexed == 0
    assert batch.files_deleted == 1
    assert store.files_table().count_rows() == 0


def test_empty_batch_is_noop(tmp_path: Path, tmp_store_dir: Path) -> None:
    indexer, _ = _make_indexer(tmp_store_dir)
    watcher = Watcher(indexer, [tmp_path])
    batch = watcher._handle_changes(set())
    assert batch.files_indexed == 0
    assert batch.files_deleted == 0
    assert watcher.stats.batches == 1


def test_running_totals_accumulate_across_batches(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    indexer, _ = _make_indexer(tmp_store_dir)
    watcher = Watcher(indexer, [tmp_path])

    a = tmp_path / "a.md"
    a.write_text("# A\nbody.\n")
    watcher._handle_changes({(Change.added, str(a))})

    b = tmp_path / "b.md"
    b.write_text("# B\nbody.\n")
    watcher._handle_changes({(Change.added, str(b))})

    assert watcher.stats.batches == 2
    assert watcher.stats.files_indexed == 2
