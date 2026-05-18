"""Feature under test: the real ``NomicTextEmbedder`` pipeline — when the
``nomic-v1.5`` ONNX weights are downloaded to the default location, the
adapter loads them, embeds text, returns L2-normalized 768-d vectors,
behaves deterministically across calls, and produces *semantically
meaningful* embeddings (sentences about transformers cluster closer to
each other than to a sentence about pasta).

This test is skipped automatically when the weights are absent; it is the
only test in the suite that exercises real model inference. It exists to
catch regressions in the BERT-style three-input feed, output pooling, and
L2-normalization that downstream code in ``LanceStore`` depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.needs_models

MODEL_DIR = Path.home() / ".lookback" / "models" / "nomic-v1.5"
MODEL_PATH = MODEL_DIR / "onnx" / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"


def _skip_if_weights_missing() -> None:
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        pytest.skip(
            f"Nomic v1.5 weights not present at {MODEL_DIR}; "
            "run `lookback models download nomic-v1.5` to enable this test."
        )


def test_real_nomic_returns_l2_normalized_768d_vector() -> None:
    _skip_if_weights_missing()
    from lookback.embed.nomic import NomicTextEmbedder

    embedder = NomicTextEmbedder(MODEL_PATH, TOKENIZER_PATH)
    vec = embedder.embed_batch(["Hello, world!"])[0]
    assert len(vec) == 768
    norm = float(np.linalg.norm(np.asarray(vec)))
    assert abs(norm - 1.0) < 1e-4, f"expected L2 norm ~1.0, got {norm}"


def test_real_nomic_is_deterministic_across_calls() -> None:
    _skip_if_weights_missing()
    from lookback.embed.nomic import NomicTextEmbedder

    embedder = NomicTextEmbedder(MODEL_PATH, TOKENIZER_PATH)
    a = embedder.embed_batch(["the same input twice"])[0]
    b = embedder.embed_batch(["the same input twice"])[0]
    # Same model, same input, same prefix — values must match bit-for-bit.
    assert a == b


def test_real_nomic_clusters_semantically_related_sentences() -> None:
    _skip_if_weights_missing()
    from lookback.embed.nomic import NomicTextEmbedder

    embedder = NomicTextEmbedder(MODEL_PATH, TOKENIZER_PATH)
    a = np.asarray(embedder.embed_batch(["Transformers use attention mechanisms."])[0])
    b = np.asarray(embedder.embed_batch(["Attention is key to transformer architectures."])[0])
    c = np.asarray(embedder.embed_batch(["I had pasta carbonara for lunch."])[0])

    # All vectors are L2-normalized, so the dot product is the cosine similarity.
    sim_related = float(a @ b)
    sim_unrelated = float(a @ c)
    assert sim_related > sim_unrelated + 0.1, (
        f"expected related sentences to be substantially closer; "
        f"related={sim_related:.4f}, unrelated={sim_unrelated:.4f}"
    )


def test_real_nomic_query_prefix_differs_from_document_prefix() -> None:
    """Embedding the same string as a query vs a document must produce
    different vectors — that's the whole point of the prefix mechanism.
    """
    _skip_if_weights_missing()
    from lookback.embed.nomic import NomicTextEmbedder

    embedder = NomicTextEmbedder(MODEL_PATH, TOKENIZER_PATH)
    doc_vec = embedder.embed_batch(["transformers"])[0]
    query_vec = embedder.embed_query("transformers")
    assert doc_vec != query_vec, (
        "query and document prefixes are supposed to produce different vectors"
    )
