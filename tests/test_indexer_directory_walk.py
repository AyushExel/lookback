"""Feature under test: ``Indexer.index_path`` recursively walks a directory.

A nested directory with mixed-modality files (markdown, python, screenshot)
should produce chunks in the right tables and report accurate stats.
Files with unknown extensions are skipped, not failed.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def _populate(root: Path) -> None:
    (root / "README.md").write_text("# Title\nSome words.\n")
    (root / "src").mkdir()
    (root / "src" / "code.py").write_text("print('hello')\n" * 5)
    (root / "shots").mkdir()
    (root / "shots" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"a" * 32)
    (root / "shots" / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"b" * 32)
    (root / "unknown.xyz").write_text("ignore me")  # unknown extension
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.md").write_text("# Hidden\nshould not be indexed.")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "pkg.md").write_text("# pkg\nignored.")


def test_directory_walk_indexes_all_supported_files(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _populate(tmp_path)
    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    stats = indexer.index_path(tmp_path)

    # README.md (1) + code.py (1) + a.png + b.png = 4 indexed files
    assert stats.files_indexed == 4
    assert stats.errors == 0
    # Text chunks: README.md (1) + code.py (1) = 2
    assert store.chunks_text_table().count_rows() == 2
    # Image chunks: 2 screenshots
    assert store.chunks_image_table().count_rows() == 2
    # files table: one row per indexed file
    assert store.files_table().count_rows() == 4


def test_hidden_and_ignored_dirs_are_skipped(tmp_path: Path, tmp_store_dir: Path) -> None:
    _populate(tmp_path)
    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    indexer.index_path(tmp_path)
    paths = {
        row["path"]
        for row in store.files_table().search().select(["path"]).limit(100).to_list()
    }
    assert not any(".hidden" in p for p in paths)
    assert not any("node_modules" in p for p in paths)
