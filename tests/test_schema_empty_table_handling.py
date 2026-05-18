"""Feature under test: empty-input handling in the Arrow conversion helpers.

When the caller passes an empty iterable, ``chunks_to_arrow`` and
``files_to_arrow`` must return an empty PyArrow table with the *correct*
schema (not a no-column table). The indexer relies on this so it can call
``table.add(empty_table)`` safely after a no-op extraction pass.
"""

from __future__ import annotations

from lookback.schema import (
    build_chunks_schema,
    chunks_to_arrow,
    files_schema,
    files_to_arrow,
)


def test_chunks_to_arrow_on_empty_iter_returns_empty_table_with_full_schema() -> None:
    table = chunks_to_arrow([], embed_dim=64)
    assert table.num_rows == 0
    assert table.schema == build_chunks_schema(64)


def test_files_to_arrow_on_empty_iter_returns_empty_table_with_full_schema() -> None:
    table = files_to_arrow([])
    assert table.num_rows == 0
    assert table.schema == files_schema()
