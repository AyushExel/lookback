"""Feature under test: bulk-adding text chunks via ``LanceStore.add_chunks``.

The store's ingest path must accept an iterable of ``ChunkRecord``, write them
in a single Arrow-batch ``add()`` call (the perf-guide-recommended bulk path),
and return the row count. Successive calls must accumulate, not overwrite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import (
    TEXT_EMBED_DIM,
    ChunkRecord,
    Modality,
)
from lookback.store.lance_store import LanceStore


def _text_chunk(i: int, file_id: str = "fA") -> ChunkRecord:
    return ChunkRecord(
        id=f"t-{i}",
        file_id=file_id,
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=i,
        text=f"chunk {i}",
        embedding=[float(i)] + [0.0] * (TEXT_EMBED_DIM - 1),
        tokens=5,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_add_returns_row_count(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    n = store.add_chunks([_text_chunk(0), _text_chunk(1), _text_chunk(2)], image=False)
    assert n == 3


def test_added_rows_are_persisted_to_disk(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_text_chunk(i) for i in range(5)], image=False)
    assert store.chunks_text_table().count_rows() == 5


def test_successive_adds_accumulate(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_text_chunk(0), _text_chunk(1)], image=False)
    store.add_chunks([_text_chunk(2)], image=False)
    assert store.chunks_text_table().count_rows() == 3


def test_empty_input_is_a_noop(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    n = store.add_chunks([], image=False)
    assert n == 0
    # The table is still created lazily on count_rows access.
    assert store.chunks_text_table().count_rows() == 0
