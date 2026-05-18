"""Feature under test: prefiltered vector search via the ``where`` parameter.

Per the LanceDB perf guide, ``prefilter=True`` applies the predicate *before*
the top-k is selected, so a filtered search returns up to ``limit`` rows that
satisfy the filter. We exercise that exact behaviour: insert chunks with
mixed ``source_kind`` values and assert that a filtered search only returns
rows matching the predicate, regardless of how close other rows are.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import TEXT_EMBED_DIM, ChunkRecord, Modality
from lookback.store.lance_store import LanceStore


def _chunk(idx: int, source_kind: str, vec: list[float]) -> ChunkRecord:
    return ChunkRecord(
        id=f"t-{idx}",
        file_id="fX",
        modality=Modality.TEXT if source_kind != "py" else Modality.CODE,
        source_kind=source_kind,
        chunk_idx=idx,
        text=f"chunk-{idx}",
        embedding=vec,
        tokens=5,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _onehot(i: int) -> list[float]:
    v = [0.0] * TEXT_EMBED_DIM
    v[i] = 1.0
    return v


def test_where_filter_returns_only_matching_rows(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    mixed = [
        _chunk(0, "markdown", _onehot(0)),
        _chunk(1, "py", _onehot(1)),
        _chunk(2, "markdown", _onehot(2)),
        _chunk(3, "py", _onehot(3)),
    ]
    store.add_chunks(mixed, image=False)

    hits = store.search_text(
        _onehot(0),
        limit=10,
        where="source_kind = 'py'",
        prefilter=True,
    )
    returned_kinds = {h["source_kind"] for h in hits}
    assert returned_kinds == {"py"}, returned_kinds
    assert {h["id"] for h in hits} == {"t-1", "t-3"}


def test_where_filter_with_no_matches_returns_empty(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_chunk(0, "markdown", _onehot(0))], image=False)
    hits = store.search_text(
        _onehot(0),
        limit=10,
        where="source_kind = 'pdf'",
        prefilter=True,
    )
    assert hits == []
