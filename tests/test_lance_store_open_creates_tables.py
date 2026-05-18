"""Feature under test: LanceStore lazily creates the three core tables.

The store should not create any tables on construction (that would race with
multiple processes opening the same path). Instead, the first access to
``chunks_text_table()``, ``chunks_image_table()``, and ``files_table()``
creates each table on demand with the schema declared in ``lookback.schema``.
"""

from __future__ import annotations

from pathlib import Path

from lookback.schema import build_chunks_schema, files_schema
from lookback.store.lance_store import (
    CHUNKS_IMAGE,
    CHUNKS_TEXT,
    FILES,
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
    LanceStore,
)


def test_no_tables_exist_before_first_access(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    assert store.table_names() == []


def test_chunks_text_table_is_created_on_first_access(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    table = store.chunks_text_table()
    assert CHUNKS_TEXT in store.table_names()
    assert table.schema == build_chunks_schema(TEXT_EMBED_DIM)


def test_chunks_image_table_is_created_on_first_access(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    table = store.chunks_image_table()
    assert CHUNKS_IMAGE in store.table_names()
    assert table.schema == build_chunks_schema(IMAGE_EMBED_DIM)


def test_files_table_is_created_on_first_access(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    table = store.files_table()
    assert FILES in store.table_names()
    assert table.schema == files_schema()


def test_repeated_access_reuses_the_same_table(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    first = store.chunks_text_table()
    second = store.chunks_text_table()
    assert first.name == second.name
    assert store.table_names().count(CHUNKS_TEXT) == 1
