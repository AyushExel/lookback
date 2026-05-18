"""Feature under test: the shape of the ``chunks`` table schema.

Verifies that ``chunks_text_schema`` and ``chunks_image_schema`` produce
PyArrow schemas with the expected columns, types, nullability, and (crucially)
the right fixed-size-list embedding dimension for each modality.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from lookback.schema import (
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
    build_chunks_schema,
    chunks_image_schema,
    chunks_text_schema,
)

EXPECTED_COLUMNS = [
    ("id", pa.string(), False),
    ("file_id", pa.string(), False),
    ("modality", pa.string(), False),
    ("source_kind", pa.string(), False),
    ("chunk_idx", pa.int32(), False),
    ("text", pa.string(), True),
    ("tokens", pa.int32(), True),
    # embedding handled separately because its type depends on dim
    ("created_at", pa.timestamp("us", tz="UTC"), False),
    ("source_mtime", pa.timestamp("us", tz="UTC"), True),
    ("meta", pa.string(), True),
]


def _assert_common_columns(schema: pa.Schema) -> None:
    for name, expected_type, nullable in EXPECTED_COLUMNS:
        field = schema.field(name)
        assert field.type == expected_type, (
            f"column {name!r}: expected type {expected_type}, got {field.type}"
        )
        assert field.nullable is nullable, (
            f"column {name!r}: expected nullable={nullable}, got {field.nullable}"
        )


def test_chunks_text_schema_has_768_dim_embedding() -> None:
    schema = chunks_text_schema()
    _assert_common_columns(schema)
    emb = schema.field("embedding")
    assert isinstance(emb.type, pa.FixedSizeListType)
    assert emb.type.list_size == TEXT_EMBED_DIM == 768
    assert emb.type.value_type == pa.float32()
    assert emb.nullable is False


def test_chunks_image_schema_has_512_dim_embedding() -> None:
    schema = chunks_image_schema()
    _assert_common_columns(schema)
    emb = schema.field("embedding")
    assert isinstance(emb.type, pa.FixedSizeListType)
    assert emb.type.list_size == IMAGE_EMBED_DIM == 512
    assert emb.type.value_type == pa.float32()


def test_build_chunks_schema_with_custom_dim() -> None:
    schema = build_chunks_schema(64)
    assert schema.field("embedding").type.list_size == 64


@pytest.mark.parametrize("bad_dim", [0, -1, -100])
def test_build_chunks_schema_rejects_nonpositive_dim(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="embed_dim must be positive"):
        build_chunks_schema(bad_dim)
