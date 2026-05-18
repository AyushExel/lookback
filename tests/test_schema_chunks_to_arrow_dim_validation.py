"""Feature under test: embedding-dimension validation in ``chunks_to_arrow``.

We want a clean ``ValueError`` naming the offending chunk and the actual vs
expected dim — not an opaque PyArrow conversion failure — when a caller
hands us a record whose embedding length doesn't match the table's fixed dim.
This matters because dim mismatches are the most common silent bug when
mixing embedders, and the perf guide reminds us that vector indexes require
fixed dimensions per column.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lookback.schema import ChunkRecord, Modality, chunks_to_arrow

EMBED_DIM = 8


def _chunk_with_dim(dim: int) -> ChunkRecord:
    return ChunkRecord(
        id="x",
        file_id="f",
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=0,
        text="t",
        embedding=[0.0] * dim,
        tokens=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_chunks_to_arrow_rejects_short_embedding() -> None:
    with pytest.raises(ValueError, match="dim 4, expected 8"):
        chunks_to_arrow([_chunk_with_dim(4)], EMBED_DIM)


def test_chunks_to_arrow_rejects_long_embedding() -> None:
    with pytest.raises(ValueError, match="dim 16, expected 8"):
        chunks_to_arrow([_chunk_with_dim(16)], EMBED_DIM)


def test_chunks_to_arrow_error_mentions_offending_chunk_id() -> None:
    bad = ChunkRecord(
        id="offender-42",
        file_id="f",
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=0,
        text="t",
        embedding=[0.0] * 3,
        tokens=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="offender-42"):
        chunks_to_arrow([bad], EMBED_DIM)
