"""Feature under test: deleting all chunks for a given ``file_id`` across both
text and image tables.

Incremental re-indexing of a changed file is done by delete-then-add (the
perf guide flags ``merge_insert`` as slower because it scans existing rows
for matches). The delete must remove every row matching ``file_id`` in both
chunk tables and must leave other files' rows untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM, ChunkRecord, Modality
from lookback.store.lance_store import LanceStore


def _text(idx: int, file_id: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"t-{file_id}-{idx}",
        file_id=file_id,
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=idx,
        text="x",
        embedding=[0.0] * TEXT_EMBED_DIM,
        tokens=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _image(idx: int, file_id: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"i-{file_id}-{idx}",
        file_id=file_id,
        modality=Modality.IMAGE,
        source_kind="png",
        chunk_idx=idx,
        text=None,
        embedding=[0.0] * IMAGE_EMBED_DIM,
        tokens=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_delete_removes_text_and_image_chunks_for_one_file(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_text(0, "A"), _text(1, "A"), _text(0, "B")], image=False)
    store.add_chunks([_image(0, "A"), _image(0, "B")], image=True)

    store.delete_chunks_by_file_id("A")

    text_ids = {
        row["id"] for row in store.chunks_text_table().search().select(["id"]).limit(100).to_list()
    }
    image_ids = {
        row["id"] for row in store.chunks_image_table().search().select(["id"]).limit(100).to_list()
    }
    assert text_ids == {"t-B-0"}
    assert image_ids == {"i-B-0"}


def test_delete_with_no_matches_is_safe(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_text(0, "A")], image=False)
    # No exception raised, no rows removed.
    store.delete_chunks_by_file_id("does-not-exist")
    assert store.chunks_text_table().count_rows() == 1
