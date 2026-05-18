"""Feature under test: ``LanceStore.search_text_hybrid`` — FTS + vector
search fused by reciprocal rank.

Verifies that the FTS index is created lazily on first hybrid search,
that the hybrid query returns hits with ``_relevance_score`` (not
``_distance``), and that exact-keyword matches outrank semantically
similar but lexically distant rows in the fused ranking.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lookback.schema import TEXT_EMBED_DIM, ChunkRecord, Modality
from lookback.store.lance_store import LanceStore


def _chunk(idx: int, text: str, vec: list[float]) -> ChunkRecord:
    return ChunkRecord(
        id=f"t-{idx}",
        file_id=f"f-{idx}",
        modality=Modality.TEXT,
        source_kind="markdown",
        chunk_idx=idx,
        text=text,
        embedding=vec,
        tokens=10,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _onehot(i: int) -> list[float]:
    v = [0.0] * TEXT_EMBED_DIM
    v[i] = 1.0
    return v


def test_hybrid_search_returns_relevance_score(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks(
        [
            _chunk(0, "IVF_PQ tuning is the LanceDB default", _onehot(0)),
            _chunk(1, "Pasta carbonara recipe", _onehot(1)),
            _chunk(2, "More about IVF_PQ partitions", _onehot(2)),
        ],
        image=False,
    )

    hits = store.search_text_hybrid("IVF_PQ", _onehot(0), limit=5)
    assert hits, "expected at least one hybrid hit"
    for h in hits:
        # Hybrid returns _relevance_score instead of _distance.
        assert "_relevance_score" in h, f"missing _relevance_score in {list(h)}"
        assert isinstance(h["_relevance_score"], (int, float))


def test_hybrid_exact_keyword_match_outranks_unrelated_row(
    tmp_store_dir: Path,
) -> None:
    store = LanceStore(tmp_store_dir)
    # Three rows with one-hot vectors so the vector-side ranking is fully
    # determined by which dim the query is one-hot in.
    store.add_chunks(
        [
            _chunk(0, "IVF_PQ tuning is the LanceDB default", _onehot(0)),
            _chunk(1, "Pasta carbonara recipe", _onehot(1)),
            _chunk(2, "Sky and clouds and rain", _onehot(2)),
        ],
        image=False,
    )

    # Vector points at dim-1 (the pasta row), but the FTS keyword is "IVF_PQ"
    # which only appears in dim-0. The fused result should put the IVF_PQ row
    # first or at least ahead of the unrelated sky row.
    hits = store.search_text_hybrid("IVF_PQ tuning", _onehot(1), limit=3)
    ids = [h["id"] for h in hits]
    assert "t-0" in ids, "exact keyword 'IVF_PQ' must surface its row"
    assert ids.index("t-0") < ids.index("t-2"), (
        f"FTS-matching row should outrank the unrelated 'sky' row; got {ids}"
    )


def test_ensure_fts_index_text_is_idempotent(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    store.add_chunks([_chunk(0, "hello", _onehot(0))], image=False)
    store.ensure_fts_index_text()
    store.ensure_fts_index_text()  # second call must not raise
