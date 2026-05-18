"""Indexer — walk roots → extract → embed → write to LanceStore.

The indexer is dependency-injected with a ``LanceStore`` plus a text and
image embedder pair plus an extractor registry. Tests use mock embedders;
the CLI plumbs the real Nomic / MobileCLIP adapters.

Incremental indexing contract:

- ``file_id`` is sha256(absolute path), stable across edits.
- On each visit, we compute the content hash; if it matches the stored row
  in the ``files`` table, the file is unchanged and we skip it.
- When content changes, we delete every chunk with that ``file_id`` from
  both chunk tables, then write the freshly extracted chunks. This is the
  perf-guide-recommended pattern over ``merge_insert``.

Batching (added in v0.1.2):

- Extraction + embedding happen per file (an unavoidable per-file cost),
  but Lance writes are *deferred* into an internal buffer and flushed in
  bulk. Each flush issues at most one delete per chunk table, one bulk
  add per chunk table, and one ``upsert_files`` call — no matter how many
  files contributed to the buffer.
- ``flush_chunk_threshold`` (default 2000) caps how many chunks pile up
  before a flush. ``flush_file_threshold`` (default 200) is the parallel
  cap on files. Whichever fires first triggers the flush.
- After every ``optimize_every_n_flushes`` flushes (default 10) the store
  is compacted (``optimize()``) to keep fragment count and version count
  bounded. A final ``optimize()`` runs at the end of ``index_path``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lookback.embed.base import ImageEmbedder, TextEmbedder
from lookback.extract.base import ExtractedChunk
from lookback.extract.registry import ExtractorRegistry
from lookback.index.chunking import estimate_tokens
from lookback.index.hashing import chunk_id, content_hash_for_path, file_id_for_path
from lookback.schema import (
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
    ChunkRecord,
    FileRecord,
    Modality,
)
from lookback.store.lance_store import LanceStore

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_DIRS = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        "target",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".idea",
        ".vscode",
    }
)

DEFAULT_FLUSH_CHUNK_THRESHOLD = 2000
DEFAULT_FLUSH_FILE_THRESHOLD = 200
DEFAULT_OPTIMIZE_EVERY_N_FLUSHES = 10


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    errors: int = 0
    errors_by_path: list[tuple[str, str]] = field(default_factory=list)
    flushes: int = 0
    optimizations: int = 0


@dataclass
class _Buffer:
    """Per-flush staging area, cleared on every ``Indexer._flush``."""

    text_records: list[ChunkRecord] = field(default_factory=list)
    image_records: list[ChunkRecord] = field(default_factory=list)
    file_records: list[FileRecord] = field(default_factory=list)
    delete_file_ids: list[str] = field(default_factory=list)

    def chunk_count(self) -> int:
        return len(self.text_records) + len(self.image_records)

    def is_empty(self) -> bool:
        return not (
            self.text_records
            or self.image_records
            or self.file_records
            or self.delete_file_ids
        )

    def clear(self) -> None:
        self.text_records.clear()
        self.image_records.clear()
        self.file_records.clear()
        self.delete_file_ids.clear()


# Callback signature: (path_just_processed, current_stats).
ProgressCallback = Callable[[Path, IndexStats], None]


class Indexer:
    def __init__(
        self,
        store: LanceStore,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        registry: ExtractorRegistry,
        *,
        max_file_bytes: int = 50 * 1024 * 1024,
        skip_hidden: bool = True,
        follow_symlinks: bool = False,
        ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
        flush_chunk_threshold: int = DEFAULT_FLUSH_CHUNK_THRESHOLD,
        flush_file_threshold: int = DEFAULT_FLUSH_FILE_THRESHOLD,
        optimize_every_n_flushes: int = DEFAULT_OPTIMIZE_EVERY_N_FLUSHES,
    ) -> None:
        if text_embedder.dim != TEXT_EMBED_DIM:
            raise ValueError(
                f"text embedder dim {text_embedder.dim} != schema dim {TEXT_EMBED_DIM}"
            )
        if image_embedder.dim != IMAGE_EMBED_DIM:
            raise ValueError(
                f"image embedder dim {image_embedder.dim} != schema dim {IMAGE_EMBED_DIM}"
            )
        if flush_chunk_threshold <= 0:
            raise ValueError("flush_chunk_threshold must be positive")
        if flush_file_threshold <= 0:
            raise ValueError("flush_file_threshold must be positive")
        if optimize_every_n_flushes <= 0:
            raise ValueError("optimize_every_n_flushes must be positive")

        self._store = store
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._registry = registry
        self._max_file_bytes = max_file_bytes
        self._skip_hidden = skip_hidden
        self._follow_symlinks = follow_symlinks
        self._ignore_dirs = ignore_dirs
        self._flush_chunk_threshold = flush_chunk_threshold
        self._flush_file_threshold = flush_file_threshold
        self._optimize_every_n_flushes = optimize_every_n_flushes
        self._buffer = _Buffer()

    def index_path(
        self,
        root: Path,
        *,
        on_file: ProgressCallback | None = None,
    ) -> IndexStats:
        """Index a file or recursively index a directory tree.

        ``on_file``, if supplied, is invoked after each file is processed
        with ``(path, stats)``. The CLI uses it to advance a progress bar;
        tests pass ``None``.
        """
        stats = IndexStats()
        root = Path(root).expanduser().resolve()
        for path in self._walk(root):
            try:
                self._process_file(path, stats)
            except Exception as exc:
                logger.warning("indexing %s failed: %s", path, exc)
                stats.errors += 1
                stats.errors_by_path.append((str(path), str(exc)))

            if on_file is not None:
                on_file(path, stats)

            if self._buffer.chunk_count() >= self._flush_chunk_threshold or (
                len(self._buffer.file_records) >= self._flush_file_threshold
            ):
                self._flush(stats)

        # Final flush + compaction.
        self._flush(stats)
        if stats.files_indexed > 0:
            try:
                self._store.optimize()
                stats.optimizations += 1
            except Exception as exc:
                logger.warning("final optimize failed: %s", exc)
        return stats

    def index_file(self, path: Path) -> IndexStats:
        """Index a single file. The watcher uses this on per-file events."""
        stats = IndexStats()
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            return stats
        if self._should_skip_file(path):
            stats.files_skipped += 1
            return stats
        try:
            self._process_file(path, stats)
        except Exception as exc:
            logger.warning("indexing %s failed: %s", path, exc)
            stats.errors += 1
            stats.errors_by_path.append((str(path), str(exc)))
        self._flush(stats)
        return stats

    def delete_path(self, path: Path) -> int:
        """Remove every chunk + the file row for ``path``. Returns chunk count."""
        path = Path(path).expanduser().resolve()
        file_id = file_id_for_path(path)
        existing = self._store.get_file(file_id)
        if not existing:
            return 0
        count = int(existing.get("chunk_count", 0))
        self._store.delete_chunks_by_file_ids([file_id])
        self._store.delete_files_by_file_id(file_id)
        return count

    def _walk(self, root: Path) -> Iterator[Path]:
        if root.is_file():
            if not self._should_skip_file(root):
                yield root
            return
        if not root.exists():
            return
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if self._should_skip_file(path, root=root):
                continue
            yield path

    def _should_skip_file(self, path: Path, *, root: Path | None = None) -> bool:
        if path.is_symlink() and not self._follow_symlinks:
            return True
        parts = path.parts
        if any(part in self._ignore_dirs for part in parts):
            return True
        if self._skip_hidden:
            try:
                rel = path.relative_to(root) if root else path
            except ValueError:
                rel = path
            for part in rel.parts:
                if part.startswith(".") and part not in {".", ".."}:
                    return True
        return False

    def _process_file(self, path: Path, stats: IndexStats) -> None:
        """Extract + embed + stage into the buffer. No Lance writes here."""
        stats.files_seen += 1

        try:
            size = path.stat().st_size
        except OSError:
            stats.files_skipped += 1
            return
        if size > self._max_file_bytes:
            stats.files_skipped += 1
            return

        extractor = self._registry.for_path(path)
        if extractor is None:
            stats.files_skipped += 1
            return

        file_id = file_id_for_path(path)
        content_hash = content_hash_for_path(path)
        existing = self._store.get_file(file_id)
        if existing and existing.get("content_hash") == content_hash:
            stats.files_unchanged += 1
            return

        extracted = extractor.extract(path)
        if not extracted:
            stats.files_skipped += 1
            return

        if existing:
            self._buffer.delete_file_ids.append(file_id)

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        text_records, image_records = self._build_records(
            extracted, file_id=file_id, mtime=mtime
        )

        self._buffer.text_records.extend(text_records)
        self._buffer.image_records.extend(image_records)
        self._buffer.file_records.append(
            FileRecord(
                file_id=file_id,
                path=str(path),
                content_hash=content_hash,
                bytes=size,
                mtime=mtime,
                last_indexed_at=datetime.now(UTC),
                extractor=extractor.__class__.__name__,
                chunk_count=len(text_records) + len(image_records),
                status="ok",
            )
        )
        stats.files_indexed += 1
        stats.chunks_written += len(text_records) + len(image_records)

    def _flush(self, stats: IndexStats) -> None:
        """Drain the buffer into Lance — bulk delete + bulk add + upsert.

        At most one delete per chunk table, one add per chunk table, one
        upsert on the files table — regardless of how many files were
        buffered. Periodic optimize keeps fragment count bounded.
        """
        if self._buffer.is_empty():
            return

        if self._buffer.delete_file_ids:
            self._store.delete_chunks_by_file_ids(self._buffer.delete_file_ids)
        self._store.add_chunks(self._buffer.text_records, image=False)
        self._store.add_chunks(self._buffer.image_records, image=True)
        if self._buffer.file_records:
            self._store.upsert_files(self._buffer.file_records)

        self._buffer.clear()
        stats.flushes += 1

        if stats.flushes % self._optimize_every_n_flushes == 0:
            try:
                self._store.optimize()
                stats.optimizations += 1
            except Exception as exc:
                logger.warning("periodic optimize failed: %s", exc)

    def _build_records(
        self,
        extracted: list[ExtractedChunk],
        *,
        file_id: str,
        mtime: datetime,
    ) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
        text_extracted: list[tuple[int, ExtractedChunk]] = []
        image_extracted: list[tuple[int, ExtractedChunk]] = []
        for idx, chunk in enumerate(extracted):
            if chunk.modality is Modality.IMAGE:
                image_extracted.append((idx, chunk))
            else:
                text_extracted.append((idx, chunk))

        text_records: list[ChunkRecord] = []
        if text_extracted:
            texts = [c.text or "" for _, c in text_extracted]
            embeddings = self._text_embedder.embed_batch(texts)
            for (idx, chunk), emb in zip(text_extracted, embeddings, strict=True):
                text_records.append(
                    ChunkRecord(
                        id=chunk_id(file_id, idx),
                        file_id=file_id,
                        modality=chunk.modality,
                        source_kind=chunk.source_kind,
                        chunk_idx=idx,
                        text=chunk.text,
                        embedding=emb,
                        tokens=estimate_tokens(chunk.text or ""),
                        source_mtime=mtime,
                        meta=chunk.meta or None,
                    )
                )

        image_records: list[ChunkRecord] = []
        if image_extracted:
            paths = [c.image_path for _, c in image_extracted if c.image_path is not None]
            if paths:
                embeddings = self._image_embedder.embed_batch(paths)
                for (idx, chunk), emb in zip(image_extracted, embeddings, strict=True):
                    image_records.append(
                        ChunkRecord(
                            id=chunk_id(file_id, idx),
                            file_id=file_id,
                            modality=chunk.modality,
                            source_kind=chunk.source_kind,
                            chunk_idx=idx,
                            text=chunk.text,
                            embedding=emb,
                            tokens=None,
                            source_mtime=mtime,
                            meta=chunk.meta or None,
                        )
                    )

        return text_records, image_records
