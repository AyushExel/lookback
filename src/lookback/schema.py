"""Schemas and record types for the Lookback Lance store.

Embeddings live in fixed-size list<float32> columns because Lance's vector
indexes (IVF_PQ, IVF_HNSW_SQ) require a fixed dimension per column. Since text
(Nomic Embed v2, 768) and image (MobileCLIP2, 512) embeddings have different
dims, we use two parallel tables — ``chunks_text`` and ``chunks_image`` — that
share every non-embedding column so the indexer and search layers can stay
modality-agnostic. See ``DESIGN.md`` § 6.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

TEXT_EMBED_DIM = 768
IMAGE_EMBED_DIM = 512


class Modality(enum.StrEnum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    URL = "url"


def build_chunks_schema(embed_dim: int) -> pa.Schema:
    """Return the chunks-table schema for a given embedding dimension.

    All columns except ``embedding`` are identical across text and image tables.
    """
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("file_id", pa.string(), nullable=False),
            pa.field("modality", pa.string(), nullable=False),
            pa.field("source_kind", pa.string(), nullable=False),
            pa.field("chunk_idx", pa.int32(), nullable=False),
            pa.field("text", pa.string(), nullable=True),
            pa.field("tokens", pa.int32(), nullable=True),
            pa.field(
                "embedding",
                pa.list_(pa.float32(), embed_dim),
                nullable=False,
            ),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("source_mtime", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("meta", pa.string(), nullable=True),
        ]
    )


def chunks_text_schema() -> pa.Schema:
    return build_chunks_schema(TEXT_EMBED_DIM)


def chunks_image_schema() -> pa.Schema:
    return build_chunks_schema(IMAGE_EMBED_DIM)


def files_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("file_id", pa.string(), nullable=False),
            pa.field("path", pa.string(), nullable=False),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("bytes", pa.int64(), nullable=False),
            pa.field("mtime", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("last_indexed_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("extractor", pa.string(), nullable=False),
            pa.field("chunk_count", pa.int32(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("error", pa.string(), nullable=True),
        ]
    )


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    id: str
    file_id: str
    modality: Modality
    source_kind: str
    chunk_idx: int
    text: str | None
    embedding: list[float]
    tokens: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_mtime: datetime | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FileRecord:
    file_id: str
    path: str
    content_hash: str
    bytes: int
    mtime: datetime
    last_indexed_at: datetime
    extractor: str
    chunk_count: int
    status: str
    error: str | None = None


def chunks_to_arrow(chunks: Iterable[ChunkRecord], embed_dim: int) -> pa.Table:
    """Convert chunk records to a single Arrow table for bulk ingest.

    Bulk Arrow conversion is the perf-guide recommended ingest path: each
    ``table.add(arrow_table)`` commits one fragment and one version, so per-row
    loops pay that overhead at every row. Always batch.
    """
    cs = list(chunks)
    schema = build_chunks_schema(embed_dim)
    if not cs:
        return schema.empty_table()

    for c in cs:
        if len(c.embedding) != embed_dim:
            raise ValueError(
                f"chunk {c.id}: embedding has dim {len(c.embedding)}, expected {embed_dim}"
            )

    columns = {
        "id": pa.array([c.id for c in cs], type=pa.string()),
        "file_id": pa.array([c.file_id for c in cs], type=pa.string()),
        "modality": pa.array([c.modality.value for c in cs], type=pa.string()),
        "source_kind": pa.array([c.source_kind for c in cs], type=pa.string()),
        "chunk_idx": pa.array([c.chunk_idx for c in cs], type=pa.int32()),
        "text": pa.array([c.text for c in cs], type=pa.string()),
        "tokens": pa.array([c.tokens for c in cs], type=pa.int32()),
        "embedding": pa.array(
            [c.embedding for c in cs],
            type=pa.list_(pa.float32(), embed_dim),
        ),
        "created_at": pa.array(
            [c.created_at for c in cs],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "source_mtime": pa.array(
            [c.source_mtime for c in cs],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "meta": pa.array(
            [json.dumps(c.meta) if c.meta is not None else None for c in cs],
            type=pa.string(),
        ),
    }
    return pa.table(columns, schema=schema)


def files_to_arrow(files: Iterable[FileRecord]) -> pa.Table:
    fs = list(files)
    schema = files_schema()
    if not fs:
        return schema.empty_table()
    columns = {
        "file_id": pa.array([f.file_id for f in fs], type=pa.string()),
        "path": pa.array([f.path for f in fs], type=pa.string()),
        "content_hash": pa.array([f.content_hash for f in fs], type=pa.string()),
        "bytes": pa.array([f.bytes for f in fs], type=pa.int64()),
        "mtime": pa.array([f.mtime for f in fs], type=pa.timestamp("us", tz="UTC")),
        "last_indexed_at": pa.array(
            [f.last_indexed_at for f in fs],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "extractor": pa.array([f.extractor for f in fs], type=pa.string()),
        "chunk_count": pa.array([f.chunk_count for f in fs], type=pa.int32()),
        "status": pa.array([f.status for f in fs], type=pa.string()),
        "error": pa.array([f.error for f in fs], type=pa.string()),
    }
    return pa.table(columns, schema=schema)
