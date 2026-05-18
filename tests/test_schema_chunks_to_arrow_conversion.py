"""Feature under test: converting ``ChunkRecord`` instances to a PyArrow table
for bulk ingest into Lance.

Verifies that ``chunks_to_arrow`` produces an Arrow table conforming to the
chunks schema with values that round-trip back to Python with no loss for
every column (including the ``meta`` JSON string, timestamps, and the
fixed-size-list embedding).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pyarrow as pa

from lookback.schema import (
    ChunkRecord,
    Modality,
    chunks_to_arrow,
)

EMBED_DIM = 8


def _make_chunk(idx: int, *, meta: dict | None = None) -> ChunkRecord:
    return ChunkRecord(
        id=f"chunk-{idx}",
        file_id=f"file-{idx // 2}",
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=idx,
        text=f"hello {idx}",
        embedding=[float(idx) + 0.1 * j for j in range(EMBED_DIM)],
        tokens=10 + idx,
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        source_mtime=datetime(2025, 12, 31, tzinfo=UTC),
        meta=meta,
    )


def test_arrow_table_has_one_row_per_chunk() -> None:
    table = chunks_to_arrow([_make_chunk(0), _make_chunk(1), _make_chunk(2)], EMBED_DIM)
    assert table.num_rows == 3


def test_arrow_table_preserves_scalar_columns() -> None:
    chunks = [_make_chunk(0), _make_chunk(1)]
    table = chunks_to_arrow(chunks, EMBED_DIM)
    assert table.column("id").to_pylist() == ["chunk-0", "chunk-1"]
    assert table.column("file_id").to_pylist() == ["file-0", "file-0"]
    assert table.column("modality").to_pylist() == ["text", "text"]
    assert table.column("source_kind").to_pylist() == ["markdown", "markdown"]
    assert table.column("chunk_idx").to_pylist() == [0, 1]
    assert table.column("text").to_pylist() == ["hello 0", "hello 1"]
    assert table.column("tokens").to_pylist() == [10, 11]


def test_arrow_table_preserves_embedding_dim_and_values() -> None:
    c = _make_chunk(3)
    table = chunks_to_arrow([c], EMBED_DIM)
    emb_type = table.schema.field("embedding").type
    assert isinstance(emb_type, pa.FixedSizeListType)
    assert emb_type.list_size == EMBED_DIM
    got = table.column("embedding").to_pylist()[0]
    assert len(got) == EMBED_DIM
    for expected, actual in zip(c.embedding, got, strict=True):
        assert abs(expected - actual) < 1e-6


def test_arrow_table_preserves_timestamps_with_utc_tz() -> None:
    table = chunks_to_arrow([_make_chunk(0)], EMBED_DIM)
    created_type = table.schema.field("created_at").type
    assert created_type == pa.timestamp("us", tz="UTC")
    val = table.column("created_at").to_pylist()[0]
    assert val == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_arrow_table_encodes_meta_as_json_string() -> None:
    chunks = [
        _make_chunk(0, meta={"page": 3, "lang": "en"}),
        _make_chunk(1, meta=None),
    ]
    table = chunks_to_arrow(chunks, EMBED_DIM)
    meta_col = table.column("meta").to_pylist()
    assert json.loads(meta_col[0]) == {"page": 3, "lang": "en"}
    assert meta_col[1] is None
