"""Feature under test: bulk-adding image chunks via ``LanceStore.add_chunks``
with ``image=True`` routing to the image table with the 512-d embedding schema.

Mirrors the text-chunk ingest test but in the image table, to catch any
modality-routing or dim-mismatch regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import IMAGE_EMBED_DIM, ChunkRecord, Modality
from lookback.store.lance_store import CHUNKS_IMAGE, CHUNKS_TEXT, LanceStore


def _image_chunk(i: int) -> ChunkRecord:
    return ChunkRecord(
        id=f"img-{i}",
        file_id="fA",
        modality=Modality.IMAGE,
        source_kind="png",
        chunk_idx=i,
        text=None,
        embedding=[float(i)] + [0.0] * (IMAGE_EMBED_DIM - 1),
        tokens=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_image_add_routes_to_image_table(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    n = store.add_chunks([_image_chunk(0), _image_chunk(1)], image=True)
    assert n == 2
    assert CHUNKS_IMAGE in store.table_names()
    assert store.chunks_image_table().count_rows() == 2


def test_image_add_does_not_write_to_text_table(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_image_chunk(0)], image=True)
    assert CHUNKS_TEXT not in store.table_names()
