"""Feature under test: ``MockTextEmbedder`` satisfies the ``TextEmbedder``
contract — declared dimension, L2-normalized outputs, deterministic vectors
for the same input string, distinct vectors for different inputs, and batch
output length equal to input length.

The indexer's correctness relies on every embedder being a pure function from
input to vector, so this test pins those guarantees for the mock.
"""

from __future__ import annotations

import math

import pytest

from lookback.embed.mock import MockTextEmbedder


def test_mock_text_embedder_reports_declared_dim() -> None:
    e = MockTextEmbedder(dim=64)
    out = e.embed_one("hello")
    assert e.dim == 64
    assert len(out) == 64


def test_mock_text_embedder_outputs_are_l2_normalized() -> None:
    e = MockTextEmbedder(dim=32)
    out = e.embed_one("hello world")
    norm = math.sqrt(sum(x * x for x in out))
    assert abs(norm - 1.0) < 1e-5
    assert e.normalized is True


def test_mock_text_embedder_is_deterministic_across_calls() -> None:
    e = MockTextEmbedder(dim=16)
    a = e.embed_one("same text")
    b = e.embed_one("same text")
    assert a == b


def test_mock_text_embedder_differs_for_different_inputs() -> None:
    e = MockTextEmbedder(dim=32)
    a = e.embed_one("alpha")
    b = e.embed_one("beta")
    assert a != b


def test_mock_text_embedder_batch_length_matches_input_length() -> None:
    e = MockTextEmbedder(dim=16)
    inputs = ["a", "b", "c", "d"]
    out = e.embed_batch(inputs)
    assert len(out) == len(inputs)
    for vec in out:
        assert len(vec) == 16


@pytest.mark.parametrize("bad_dim", [0, -1, -100])
def test_mock_text_embedder_rejects_nonpositive_dim(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        MockTextEmbedder(dim=bad_dim)
