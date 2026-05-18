"""Feature under test: ``LanceStore.search_text`` returns the top-k chunks by
vector similarity using cosine distance (Nomic-friendly), respecting
``limit`` and projecting the default column set.

This exercises the brute-force flat-search path: we don't build a vector
index in v0 unit tests because IVF_PQ requires far more training rows than
a handful of fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import TEXT_EMBED_DIM, ChunkRecord, Modality
from lookback.store.lance_store import DEFAULT_SEARCH_PROJECTION, LanceStore


def _chunk(idx: int, vec: list[float]) -> ChunkRecord:
    assert len(vec) == TEXT_EMBED_DIM
    return ChunkRecord(
        id=f"t-{idx}",
        file_id="fX",
        modality=Modality.TEXT,
        source_kind="markdown",
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


def test_top_hit_is_the_chunk_with_closest_embedding(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    chunks = [_chunk(i, _onehot(i)) for i in range(5)]
    store.add_chunks(chunks, image=False)

    # Query vector very close to chunk #3 (1.0 in dim 3) — cosine should rank it first.
    query = _onehot(3)
    hits = store.search_text(query, limit=3)

    assert len(hits) == 3
    assert hits[0]["id"] == "t-3"


def test_limit_caps_returned_rows(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_chunk(i, _onehot(i)) for i in range(5)], image=False)
    hits = store.search_text(_onehot(0), limit=2)
    assert len(hits) == 2


def test_default_projection_includes_design_columns(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_chunk(0, _onehot(0))], image=False)
    hits = store.search_text(_onehot(0), limit=1)
    assert hits, "expected at least one hit"
    for col in DEFAULT_SEARCH_PROJECTION:
        assert col in hits[0], f"missing projected column: {col}"
    # Embedding deliberately excluded from default projection to avoid bloat.
    assert "embedding" not in hits[0]
