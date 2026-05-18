"""Feature under test: writing and counting rows in the ``files`` table.

``files`` carries file-level metadata used for incremental indexing
(content hashes, mtimes, last-indexed markers). It has the same bulk-add
contract as the chunk tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import FileRecord
from lookback.store.lance_store import FILES, LanceStore


def _file(idx: int, status: str = "ok") -> FileRecord:
    return FileRecord(
        file_id=f"file-{idx}",
        path=f"/tmp/x/{idx}.md",
        content_hash=f"hash-{idx}",
        bytes=100 + idx,
        mtime=datetime(2026, 1, 1, tzinfo=UTC),
        last_indexed_at=datetime(2026, 1, 2, tzinfo=UTC),
        extractor="markdown",
        chunk_count=3,
        status=status,
        error=None,
    )


def test_add_files_returns_row_count(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    n = store.add_files([_file(0), _file(1), _file(2)])
    assert n == 3


def test_files_table_count_matches_after_add(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_files([_file(0), _file(1)])
    assert store.files_table().count_rows() == 2
    assert FILES in store.table_names()


def test_empty_files_add_is_noop(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    assert store.add_files([]) == 0
