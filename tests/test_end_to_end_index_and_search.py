"""Feature under test: full pipeline integration.

Builds a fixture directory with markdown, code, and screenshot files. Indexes
it via the indexer (with mock embedders). Searches for an exact chunk-text
match and asserts the matching chunk comes back as the top hit (the mock
embedder is deterministic, so embedding the same text twice gives the same
vector — cosine distance 0). This proves the layers compose end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def _build_fixture_tree(root: Path) -> None:
    (root / "notes").mkdir()
    (root / "notes" / "lance.md").write_text(
        "# LanceDB perf\nUse IVF_PQ for general workloads.\n"
    )
    (root / "notes" / "rust.md").write_text(
        "# Rust\nOwnership and lifetimes are the core ideas.\n"
    )
    (root / "code").mkdir()
    (root / "code" / "hello.py").write_text(
        "def greet(name):\n    return f'hello {name}'\n"
    )
    (root / "shots").mkdir()
    (root / "shots" / "screenshot.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"abc" * 32
    )


def test_index_then_search_returns_matching_chunk_as_top_hit(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _build_fixture_tree(tmp_path)

    store = LanceStore(tmp_store_dir)
    text_embedder = MockTextEmbedder(dim=TEXT_EMBED_DIM)
    indexer = Indexer(
        store=store,
        text_embedder=text_embedder,
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    stats = indexer.index_path(tmp_path)
    assert stats.files_indexed == 4
    assert stats.errors == 0

    stored_chunks = (
        store.chunks_text_table()
        .search()
        .select(["id", "text", "source_kind"])
        .limit(100)
        .to_list()
    )
    assert stored_chunks, "expected indexed chunks"

    # Pick any stored chunk and re-embed its exact text — distance should be ~0.
    target = next(c for c in stored_chunks if c["source_kind"] == "markdown")
    query_vec = text_embedder.embed_query(target["text"])
    hits = store.search_text(query_vec, limit=3)
    assert hits, "expected at least one search hit"
    assert hits[0]["id"] == target["id"], (
        f"expected exact-text query to return the source chunk first, "
        f"got {hits[0]['id']} vs {target['id']}"
    )


def test_modality_filter_narrows_search_results(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _build_fixture_tree(tmp_path)

    store = LanceStore(tmp_store_dir)
    text_embedder = MockTextEmbedder(dim=TEXT_EMBED_DIM)
    indexer = Indexer(
        store=store,
        text_embedder=text_embedder,
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    indexer.index_path(tmp_path)

    query_vec = text_embedder.embed_query("anything")
    code_only = store.search_text(
        query_vec, limit=20, where="modality = 'code'"
    )
    text_only = store.search_text(
        query_vec, limit=20, where="modality = 'text'"
    )
    assert all(h["modality"] == "code" for h in code_only)
    assert all(h["modality"] == "text" for h in text_only)
    # Neither should be empty given the fixture tree.
    assert code_only
    assert text_only


def test_reindexing_unchanged_tree_writes_no_new_chunks(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    _build_fixture_tree(tmp_path)

    store = LanceStore(tmp_store_dir)
    indexer = Indexer(
        store=store,
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
    )
    indexer.index_path(tmp_path)
    first_text_count = store.chunks_text_table().count_rows()
    first_image_count = store.chunks_image_table().count_rows()

    second = indexer.index_path(tmp_path)

    assert second.files_unchanged == 4
    assert second.files_indexed == 0
    assert store.chunks_text_table().count_rows() == first_text_count
    assert store.chunks_image_table().count_rows() == first_image_count
