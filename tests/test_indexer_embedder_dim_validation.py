"""Feature under test: the indexer refuses to start when an embedder reports
a dimension that doesn't match the Lance schema. Catching this in the
constructor (rather than at first write) gives a clear error before any data
is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore


def test_text_embedder_with_wrong_dim_raises(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    with pytest.raises(ValueError, match=f"text embedder dim 7 != schema dim {TEXT_EMBED_DIM}"):
        Indexer(
            store=store,
            text_embedder=MockTextEmbedder(dim=7),
            image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
            registry=default_registry(),
        )


def test_image_embedder_with_wrong_dim_raises(tmp_store_dir: Path) -> None:
    store = LanceStore(tmp_store_dir)
    with pytest.raises(ValueError, match=f"image embedder dim 7 != schema dim {IMAGE_EMBED_DIM}"):
        Indexer(
            store=store,
            text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
            image_embedder=MockImageEmbedder(dim=7),
            registry=default_registry(),
        )
