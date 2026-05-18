"""Feature under test: ``MobileCLIPImageEmbedder`` honours ``batch_size``
by splitting ``embed_batch`` into bounded ONNX calls.

Mirrors the Nomic mini-batching test for the image side. The image
embedder's per-batch intermediate is smaller (output is 512-d, no
``last_hidden_state``-style intermediate) but preprocessing materializes
``(N, 3, 256, 256)`` float32 — at the default ``batch_size=8`` that's
6 MB regardless of how many screenshots one file produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.embed.mobileclip import MobileCLIPImageEmbedder


def _stub_loaded(embedder: MobileCLIPImageEmbedder) -> None:
    embedder._session = object()


def test_embed_batch_calls_session_once_per_batch_size_slice(tmp_path: Path) -> None:
    embedder = MobileCLIPImageEmbedder(tmp_path / "fake.onnx", batch_size=4)
    _stub_loaded(embedder)
    call_sizes: list[int] = []

    def fake_batch(paths: list[Path]) -> list[list[float]]:
        call_sizes.append(len(paths))
        return [[0.0] * embedder.dim for _ in paths]

    embedder._embed_one_batch = fake_batch  # type: ignore[method-assign]

    fakes = [tmp_path / f"img_{i}.png" for i in range(10)]
    result = embedder.embed_batch(fakes)
    assert len(result) == 10
    assert call_sizes == [4, 4, 2]


def test_embed_batch_on_empty_input_does_not_load_or_call_session(
    tmp_path: Path,
) -> None:
    embedder = MobileCLIPImageEmbedder(tmp_path / "no-such-model.onnx")
    assert embedder.embed_batch([]) == []


@pytest.mark.parametrize("bad_batch_size", [0, -1])
def test_constructor_rejects_nonpositive_batch_size(
    bad_batch_size: int, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        MobileCLIPImageEmbedder(tmp_path / "fake.onnx", batch_size=bad_batch_size)
