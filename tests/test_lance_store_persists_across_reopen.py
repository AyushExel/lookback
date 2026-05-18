"""Feature under test: data persists across separate ``LanceStore`` instances
on the same path.

Closing the Python object should not lose data, and a fresh ``LanceStore``
pointed at the same directory must see every previously-written row. This is
the bedrock contract we depend on for incremental indexing and for running
the indexer in a separate process from the search/MCP server.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import TEXT_EMBED_DIM, ChunkRecord, Modality
from lookback.store.lance_store import LanceStore


def _chunk(i: int) -> ChunkRecord:
    return ChunkRecord(
        id=f"t-{i}",
        file_id="f1",
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=i,
        text=f"chunk {i}",
        embedding=[0.0] * TEXT_EMBED_DIM,
        tokens=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_chunks_are_visible_after_reopen(tmp_store_dir: Path) -> None:
    first = LanceStore(tmp_store_dir)
    first.add_chunks([_chunk(0), _chunk(1), _chunk(2)], image=False)
    del first

    second = LanceStore(tmp_store_dir)
    assert second.chunks_text_table().count_rows() == 3
    ids = {
        row["id"] for row in second.chunks_text_table().search().select(["id"]).limit(100).to_list()
    }
    assert ids == {"t-0", "t-1", "t-2"}
