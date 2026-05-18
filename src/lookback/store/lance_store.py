"""LanceStore — local persistence for Lookback, tuned per the LanceDB perf guide.

Design notes (see ``DESIGN.md`` § 7):

- **Bulk writes only.** Every write path goes through ``table.add(arrow_table)``
  with a multi-row Arrow table. Per-row inserts commit one fragment and one
  version per call — they will tank ingest perf at any real scale.
- **Scalar indexes are created up front** (bitmap for low-cardinality columns
  like ``modality`` / ``source_kind`` / ``status`` / ``extractor``; btree for
  IDs, paths, and timestamps) so filtered searches stay fast.
- **Vector indexes are NOT auto-built.** Call ``build_vector_index()`` after
  bulk loading; rebuilding after every small write thrashes, and the guide
  recommends rebuilding only after large update batches.
- **Searches always project (``.select``) and limit (``.limit``).** We never
  call ``to_pandas()`` or ``to_arrow()`` on tables.
- **Distance metrics:** ``cosine`` for text (Nomic Embed v2 is not L2
  normalized) and ``dot`` for image (MobileCLIP2 is L2 normalized — ``dot`` is
  fastest when inputs are normalized).
- **Filtered search uses ``prefilter=True``** by default; the guide notes that
  post-filter applies the predicate after the top-k is picked, so filtered
  candidates can fall under the requested ``limit``.
- **Re-indexing changed files: delete-then-add by ``file_id``.** ``merge_insert``
  is slower per the guide because it scans existing rows for matches.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import lancedb
import pyarrow as pa

from lookback.schema import (
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
    ChunkRecord,
    FileRecord,
    build_chunks_schema,
    chunks_to_arrow,
    files_schema,
    files_to_arrow,
)

logger = logging.getLogger(__name__)

CHUNKS_TEXT = "chunks_text"
CHUNKS_IMAGE = "chunks_image"
FILES = "files"

DEFAULT_SEARCH_PROJECTION = [
    "id",
    "file_id",
    "modality",
    "source_kind",
    "chunk_idx",
    "text",
    "created_at",
    "source_mtime",
    "meta",
    "_distance",
]

# Columns we want callers to see from a hybrid result. Note: lance's hybrid
# query path doesn't accept ``.select([...])`` because the FTS subquery and
# vector subquery have different schemas — projection fails the validator.
# We drop the embedding column in Python after the round trip instead.
HYBRID_RETURNED_COLUMNS = (
    "id",
    "file_id",
    "modality",
    "source_kind",
    "chunk_idx",
    "text",
    "created_at",
    "source_mtime",
    "meta",
    "_relevance_score",
)

_CHUNK_BITMAP_COLS = ("modality", "source_kind")
_CHUNK_BTREE_COLS = ("id", "file_id", "created_at", "source_mtime")
_FILE_BITMAP_COLS = ("status", "extractor")
_FILE_BTREE_COLS = ("file_id", "path", "content_hash", "mtime", "last_indexed_at")


class LanceStore:
    """Thin, perf-guide-conformant wrapper around a local LanceDB connection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.path))
        # Table handles are cached per-store so we don't pay a manifest re-read
        # on every ``files_table()`` / ``chunks_text_table()`` access — the
        # indexer calls those once per file, and on a 10k-file walk that's
        # what dominates wall-clock time. Writes through a cached handle
        # remain visible to subsequent reads through the same handle.
        self._table_cache: dict[str, Any] = {}

    def table_names(self) -> list[str]:
        """Return the names of every table currently in the store.

        ``lancedb.connect(...).list_tables()`` evolved between versions:
        - Older releases returned an iterable of name strings.
        - 0.30 returns a paginated response object with a ``tables`` field.
        We handle both, plus any future variants that hand back a dict.
        """
        result = self._db.list_tables()
        if hasattr(result, "tables"):
            return list(result.tables)
        if isinstance(result, dict):
            return list(result.get("tables", []))
        items = list(result)
        return [name for name in items if isinstance(name, str)]

    def _open_or_create(
        self,
        name: str,
        schema: pa.Schema,
        *,
        bitmap_cols: tuple[str, ...] = (),
        btree_cols: tuple[str, ...] = (),
    ) -> Any:
        cached = self._table_cache.get(name)
        if cached is not None:
            return cached
        if name in self.table_names():
            table = self._db.open_table(name)
            self._table_cache[name] = table
            return table
        table = self._db.create_table(name, schema=schema)
        for col in bitmap_cols:
            try:
                table.create_scalar_index(col, index_type="BITMAP")
            except Exception as exc:
                logger.debug("bitmap index deferred on %s.%s: %s", name, col, exc)
        for col in btree_cols:
            try:
                table.create_scalar_index(col, index_type="BTREE")
            except Exception as exc:
                logger.debug("btree index deferred on %s.%s: %s", name, col, exc)
        self._table_cache[name] = table
        return table

    def chunks_text_table(self) -> Any:
        return self._open_or_create(
            CHUNKS_TEXT,
            build_chunks_schema(TEXT_EMBED_DIM),
            bitmap_cols=_CHUNK_BITMAP_COLS,
            btree_cols=_CHUNK_BTREE_COLS,
        )

    def chunks_image_table(self) -> Any:
        return self._open_or_create(
            CHUNKS_IMAGE,
            build_chunks_schema(IMAGE_EMBED_DIM),
            bitmap_cols=_CHUNK_BITMAP_COLS,
            btree_cols=_CHUNK_BTREE_COLS,
        )

    def files_table(self) -> Any:
        return self._open_or_create(
            FILES,
            files_schema(),
            bitmap_cols=_FILE_BITMAP_COLS,
            btree_cols=_FILE_BTREE_COLS,
        )

    def add_chunks(
        self,
        chunks: Iterable[ChunkRecord],
        *,
        image: bool,
    ) -> int:
        """Bulk add. Returns the number of rows written.

        ``image=True`` writes to ``chunks_image``; otherwise ``chunks_text``.
        Empty input is a no-op.
        """
        cs = list(chunks)
        if not cs:
            return 0
        dim = IMAGE_EMBED_DIM if image else TEXT_EMBED_DIM
        table = self.chunks_image_table() if image else self.chunks_text_table()
        arrow_table = chunks_to_arrow(cs, dim)
        table.add(arrow_table)
        return arrow_table.num_rows

    def add_files(self, files: Iterable[FileRecord]) -> int:
        fs = list(files)
        if not fs:
            return 0
        arrow_table = files_to_arrow(fs)
        self.files_table().add(arrow_table)
        return arrow_table.num_rows

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """Remove every chunk (text and image) belonging to ``file_id``."""
        self.delete_chunks_by_file_ids([file_id])

    def delete_chunks_by_file_ids(self, file_ids: Iterable[str]) -> None:
        """Bulk-remove every chunk belonging to *any* of the listed file ids.

        Single ``IN (...)`` delete per chunk table — a batched re-index pass
        of N changed files commits two deletes instead of 2N, matching the
        perf-guide guidance against per-row writes.
        """
        ids = list(dict.fromkeys(file_ids))  # de-dupe while preserving order
        if not ids:
            return
        safe = ", ".join(
            f"'{fid.replace(chr(39), chr(39) + chr(39))}'" for fid in ids
        )
        predicate = f"file_id IN ({safe})"
        for tbl in (self.chunks_text_table(), self.chunks_image_table()):
            tbl.delete(predicate)

    def upsert_files(self, files: Iterable[FileRecord]) -> int:
        """Replace any existing rows with matching ``file_id`` and add the new ones.

        ``merge_insert`` would be the obvious tool, but the perf guide flags it
        as slower than delete-then-add (it scans existing rows for matches).
        Our scalar btree on ``file_id`` makes the delete fast.
        """
        fs = list(files)
        if not fs:
            return 0
        ids = {f.file_id for f in fs}
        safe_ids = ", ".join(f"'{fid.replace(chr(39), chr(39) + chr(39))}'" for fid in ids)
        self.files_table().delete(f"file_id IN ({safe_ids})")
        return self.add_files(fs)

    def get_files_paths(self, file_ids: Iterable[str]) -> dict[str, str]:
        """Resolve ``file_id -> absolute path`` for a batch of ids.

        Used by the CLI search renderer to turn raw hits into clickable
        ``file://`` links. One scan of the files table per query, even
        when there are many hits.
        """
        ids = list(dict.fromkeys(file_ids))
        if not ids:
            return {}
        safe = ", ".join(
            f"'{fid.replace(chr(39), chr(39) + chr(39))}'" for fid in ids
        )
        rows = (
            self.files_table()
            .search()
            .where(f"file_id IN ({safe})")
            .select(["file_id", "path"])
            .limit(len(ids))
            .to_list()
        )
        return {r["file_id"]: r["path"] for r in rows}

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        """Return the single ``files`` row matching ``file_id`` or None."""
        safe = file_id.replace("'", "''")
        rows = (
            self.files_table()
            .search()
            .where(f"file_id = '{safe}'")
            .limit(1)
            .to_list()
        )
        return rows[0] if rows else None

    def delete_files_by_file_id(self, file_id: str) -> None:
        """Remove the row in ``files`` matching ``file_id`` (no-op if absent)."""
        safe = file_id.replace("'", "''")
        self.files_table().delete(f"file_id = '{safe}'")

    def search_text(
        self,
        query_vec: list[float],
        *,
        limit: int = 20,
        where: str | None = None,
        prefilter: bool = True,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._search(
            self.chunks_text_table(),
            query_vec,
            distance="cosine",
            limit=limit,
            where=where,
            prefilter=prefilter,
            columns=columns,
        )

    def ensure_fts_index_text(self) -> None:
        """Create the FTS index on ``chunks_text.text`` if it isn't there yet.

        Per the perf guide we leave ``with_position=False`` and
        ``remove_stop_words=True`` — phrase-search flags inflate index size
        and build time and we don't need them for ranking.
        """
        table = self.chunks_text_table()
        try:
            indexes = table.list_indices()
        except Exception:  # pragma: no cover - older API fallback
            indexes = []
        for idx in indexes:
            if getattr(idx, "columns", None) == ["text"]:
                return
        try:
            table.create_fts_index(
                "text",
                with_position=False,
                remove_stop_words=True,
                replace=True,
            )
        except Exception as exc:
            logger.warning("creating FTS index on chunks_text.text failed: %s", exc)

    def search_text_hybrid(
        self,
        query_text: str,
        query_vec: list[float],
        *,
        limit: int = 20,
        where: str | None = None,
        prefilter: bool = True,
        columns: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """FTS + vector search on ``chunks_text``, fused by reciprocal rank.

        Returned rows carry ``_relevance_score`` (higher = better) instead
        of ``_distance``. We drop the ``embedding`` column post-hoc since
        lance's hybrid path doesn't accept a server-side projection
        (vector- and FTS-subquery schemas differ, projection validation
        fails). Bandwidth impact is small at typical ``limit`` values.
        """
        self.ensure_fts_index_text()
        table = self.chunks_text_table()
        q = (
            table.search(query_type="hybrid")
            .vector(query_vec)
            .text(query_text)
            .limit(limit)
        )
        if where:
            q = q.where(where, prefilter=prefilter)

        keep = set(columns) if columns is not None else set(HYBRID_RETURNED_COLUMNS)
        # Always keep the score; callers expect it.
        keep.add("_relevance_score")
        rows = q.to_list()
        return [{k: v for k, v in row.items() if k in keep} for row in rows]

    def search_image(
        self,
        query_vec: list[float],
        *,
        limit: int = 20,
        where: str | None = None,
        prefilter: bool = True,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._search(
            self.chunks_image_table(),
            query_vec,
            distance="dot",
            limit=limit,
            where=where,
            prefilter=prefilter,
            columns=columns,
        )

    def _search(
        self,
        table: Any,
        query_vec: list[float],
        *,
        distance: Literal["cosine", "dot", "l2"],
        limit: int,
        where: str | None,
        prefilter: bool,
        columns: list[str] | None,
    ) -> list[dict[str, Any]]:
        cols = columns if columns is not None else DEFAULT_SEARCH_PROJECTION
        q = table.search(query_vec).distance_type(distance).limit(limit).select(cols)
        if where:
            q = q.where(where, prefilter=prefilter)
        return q.to_list()

    def build_vector_index(
        self,
        *,
        image: bool,
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
    ) -> None:
        """Build the IVF_PQ vector index on the chunks table.

        Call this after a bulk load; rebuilding after small writes thrashes.
        Default parameters let LanceDB pick partition/sub-vector counts from
        the dataset size, which is correct unless you have a measured reason
        to override.
        """
        table = self.chunks_image_table() if image else self.chunks_text_table()
        kwargs: dict[str, Any] = {
            "metric": "dot" if image else "cosine",
            "index_type": "IVF_PQ",
        }
        if num_partitions is not None:
            kwargs["num_partitions"] = num_partitions
        if num_sub_vectors is not None:
            kwargs["num_sub_vectors"] = num_sub_vectors
        table.create_index(**kwargs)

    def optimize(self, *, cleanup_older_than_days: int = 7) -> None:
        """Compact fragments and prune old versions.

        Run after bulk index passes. The guide's default retention window is 7
        days; old versions consume storage but offer no read-side benefit
        once superseded.
        """
        cleanup = timedelta(days=cleanup_older_than_days)
        for tbl in (self.chunks_text_table(), self.chunks_image_table(), self.files_table()):
            try:
                tbl.optimize(cleanup_older_than=cleanup)
            except Exception as exc:
                logger.warning("optimize failed on %s: %s", tbl.name, exc)

    def stats(self) -> dict[str, int]:
        return {
            CHUNKS_TEXT: self.chunks_text_table().count_rows(),
            CHUNKS_IMAGE: self.chunks_image_table().count_rows(),
            FILES: self.files_table().count_rows(),
        }
