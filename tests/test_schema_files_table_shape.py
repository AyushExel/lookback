"""Feature under test: the shape of the ``files`` table schema.

Verifies that ``files_schema`` declares every column needed for incremental
indexing — id, path, content hash, byte size, mtime, last-indexed marker,
extractor tag, chunk count, status, and an optional error string — at the
right type and nullability.
"""

from __future__ import annotations

import pyarrow as pa

from lookback.schema import files_schema

EXPECTED = {
    "file_id": (pa.string(), False),
    "path": (pa.string(), False),
    "content_hash": (pa.string(), False),
    "bytes": (pa.int64(), False),
    "mtime": (pa.timestamp("us", tz="UTC"), False),
    "last_indexed_at": (pa.timestamp("us", tz="UTC"), False),
    "extractor": (pa.string(), False),
    "chunk_count": (pa.int32(), False),
    "status": (pa.string(), False),
    "error": (pa.string(), True),
}


def test_files_schema_columns_match_design() -> None:
    schema = files_schema()
    assert set(schema.names) == set(EXPECTED.keys()), (
        f"unexpected columns: {set(schema.names) ^ set(EXPECTED.keys())}"
    )
    for name, (expected_type, nullable) in EXPECTED.items():
        field = schema.field(name)
        assert field.type == expected_type, (
            f"column {name!r}: expected type {expected_type}, got {field.type}"
        )
        assert field.nullable is nullable, (
            f"column {name!r}: expected nullable={nullable}, got {field.nullable}"
        )
