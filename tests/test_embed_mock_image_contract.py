"""Feature under test: ``MockImageEmbedder`` satisfies the ``ImageEmbedder``
contract — declared dimension, L2-normalized outputs, deterministic vectors
for the same image bytes, distinct vectors for different image bytes, and
batch output length equal to input length.
"""

from __future__ import annotations

import math
from pathlib import Path

from lookback.embed.mock import MockImageEmbedder


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_mock_image_embedder_reports_declared_dim(tmp_path: Path) -> None:
    e = MockImageEmbedder(dim=64)
    img = _write_bytes(tmp_path, "a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    out = e.embed_one(img)
    assert e.dim == 64
    assert len(out) == 64


def test_mock_image_embedder_outputs_are_l2_normalized(tmp_path: Path) -> None:
    e = MockImageEmbedder(dim=32)
    img = _write_bytes(tmp_path, "a.png", b"any bytes here")
    out = e.embed_one(img)
    norm = math.sqrt(sum(x * x for x in out))
    assert abs(norm - 1.0) < 1e-5
    assert e.normalized is True


def test_mock_image_embedder_is_deterministic_across_calls(tmp_path: Path) -> None:
    e = MockImageEmbedder(dim=16)
    img = _write_bytes(tmp_path, "x.png", b"fixed content")
    a = e.embed_one(img)
    b = e.embed_one(img)
    assert a == b


def test_mock_image_embedder_differs_for_different_bytes(tmp_path: Path) -> None:
    e = MockImageEmbedder(dim=32)
    a = _write_bytes(tmp_path, "a.png", b"AAAA")
    b = _write_bytes(tmp_path, "b.png", b"BBBB")
    assert e.embed_one(a) != e.embed_one(b)


def test_mock_image_embedder_batch_length_matches_input_length(tmp_path: Path) -> None:
    e = MockImageEmbedder(dim=16)
    paths = [_write_bytes(tmp_path, f"{i}.png", bytes([i] * 8)) for i in range(4)]
    out = e.embed_batch(paths)
    assert len(out) == len(paths)
    for v in out:
        assert len(v) == 16
