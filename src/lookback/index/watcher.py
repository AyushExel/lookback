"""File watcher — re-indexes on the fly when files change under a root.

Uses ``watchfiles`` (Rust-backed via notify) so we don't pay a Python event
loop per filesystem event. Events arrive in batches per debounce window;
the watcher resolves each batch into the unique set of files to (re)index
and the set of files to remove from the store.

The change-handling logic is a pure method, ``_handle_changes``, so tests
can drive it without standing up the actual filesystem watch.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from watchfiles import Change, watch

from lookback.index.indexer import Indexer

logger = logging.getLogger(__name__)


@dataclass
class WatchStats:
    """Per-batch summary, useful for CLI output and tests."""

    batches: int = 0
    files_indexed: int = 0
    files_deleted: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    errors: int = 0
    error_paths: list[str] = field(default_factory=list)


class Watcher:
    def __init__(
        self,
        indexer: Indexer,
        roots: Iterable[Path],
        *,
        debounce_ms: int = 400,
        recursive: bool = True,
    ) -> None:
        self._indexer = indexer
        self._roots = [Path(r).expanduser().resolve() for r in roots]
        self._debounce_ms = debounce_ms
        self._recursive = recursive
        self.stats = WatchStats()

    def _handle_changes(self, changes: Iterable[tuple[Change, str]]) -> WatchStats:
        """Process one batch of FS events and return per-batch deltas.

        We also accumulate into ``self.stats`` so the long-running watch
        can report a running total.
        """
        added_or_modified: set[Path] = set()
        deleted: set[Path] = set()

        for change_type, path_str in changes:
            p = Path(path_str)
            if change_type == Change.deleted:
                deleted.add(p)
                # A delete-then-recreate within the same batch should still
                # re-index, so don't blanket-drop from the modified set yet.
            else:
                added_or_modified.add(p)

        # If a path is both modified and deleted in the same batch, trust the
        # last filesystem state — check what actually exists on disk.
        for p in list(added_or_modified):
            if not p.exists():
                added_or_modified.discard(p)
                deleted.add(p)

        batch = WatchStats(batches=1)

        for p in added_or_modified:
            try:
                fs = self._indexer.index_file(p)
                batch.files_indexed += fs.files_indexed
                batch.chunks_written += fs.chunks_written
                batch.errors += fs.errors
                batch.error_paths.extend(path for path, _ in fs.errors_by_path)
            except Exception as exc:
                logger.warning("watcher: indexing %s failed: %s", p, exc)
                batch.errors += 1
                batch.error_paths.append(str(p))

        for p in deleted:
            try:
                removed = self._indexer.delete_path(p)
                if removed:
                    batch.files_deleted += 1
                    batch.chunks_deleted += removed
            except Exception as exc:
                logger.warning("watcher: delete %s failed: %s", p, exc)
                batch.errors += 1
                batch.error_paths.append(str(p))

        # Accumulate into running totals.
        self.stats.batches += batch.batches
        self.stats.files_indexed += batch.files_indexed
        self.stats.files_deleted += batch.files_deleted
        self.stats.chunks_written += batch.chunks_written
        self.stats.chunks_deleted += batch.chunks_deleted
        self.stats.errors += batch.errors
        self.stats.error_paths.extend(batch.error_paths)
        return batch

    def run(self) -> None:
        """Blocking watch loop. Stop with Ctrl-C."""
        logger.info("watching %s", [str(r) for r in self._roots])
        for changes in watch(
            *[str(r) for r in self._roots],
            step=self._debounce_ms,
            recursive=self._recursive,
        ):
            self._handle_changes(changes)
