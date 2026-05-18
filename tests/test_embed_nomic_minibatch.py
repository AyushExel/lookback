"""Feature under test: ``NomicTextEmbedder`` honours ``batch_size`` by
splitting ``embed_batch`` into bounded ONNX calls.

This is the v0.1.2 fix for the memory bomb that hit when indexing
real-world folders. A single huge file (a PDF or log producing thousands
of chunks) was previously embedded in one ONNX call, producing a
``(N, 512, 768)`` float32 ``last_hidden_state`` tensor that scaled
linearly with chunk count — into multi-GB territory for large files.

The fix: chunk the input list into ``batch_size`` slices before calling
the ONNX session. Peak memory becomes a function of ``batch_size``, not
of how many chunks a single file produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.embed.nomic import NomicTextEmbedder


def _stub_loaded(embedder: NomicTextEmbedder) -> None:
    """Mark the embedder as 'loaded' without actually opening any files.

    The internal ``_embed_one_batch`` is what we'll replace per test —
    we just need to bypass the ``_ensure_loaded`` file-existence checks.
    """
    embedder._session = object()
    embedder._tokenizer = object()


def test_embed_batch_calls_session_once_per_batch_size_slice(tmp_path: Path) -> None:
    embedder = NomicTextEmbedder(
        tmp_path / "fake.onnx", tmp_path / "fake.tok", batch_size=4
    )
    _stub_loaded(embedder)
    call_sizes: list[int] = []

    def fake_batch(texts: list[str], *, prefix: str) -> list[list[float]]:
        call_sizes.append(len(texts))
        return [[0.0] * embedder.dim for _ in texts]

    embedder._embed_one_batch = fake_batch  # type: ignore[method-assign]

    result = embedder.embed_batch(["x"] * 10)
    assert len(result) == 10
    # 10 inputs with batch_size=4 → slices of 4, 4, 2.
    assert call_sizes == [4, 4, 2]


def test_embed_batch_with_input_smaller_than_batch_size_is_one_call(
    tmp_path: Path,
) -> None:
    embedder = NomicTextEmbedder(
        tmp_path / "fake.onnx", tmp_path / "fake.tok", batch_size=32
    )
    _stub_loaded(embedder)
    call_sizes: list[int] = []

    def fake_batch(texts: list[str], *, prefix: str) -> list[list[float]]:
        call_sizes.append(len(texts))
        return [[0.0] * embedder.dim for _ in texts]

    embedder._embed_one_batch = fake_batch  # type: ignore[method-assign]
    embedder.embed_batch(["x"] * 5)
    assert call_sizes == [5]


def test_embed_batch_on_empty_input_does_not_load_or_call_session(
    tmp_path: Path,
) -> None:
    embedder = NomicTextEmbedder(
        tmp_path / "no-such-model.onnx", tmp_path / "no-such-tok.json"
    )
    # No stubbing: if `_ensure_loaded` ran we'd hit FileNotFoundError.
    assert embedder.embed_batch([]) == []


def test_embed_query_uses_query_prefix_in_one_batch_call(tmp_path: Path) -> None:
    embedder = NomicTextEmbedder(
        tmp_path / "fake.onnx", tmp_path / "fake.tok", batch_size=4
    )
    _stub_loaded(embedder)
    seen_prefix: list[str] = []

    def fake_batch(texts: list[str], *, prefix: str) -> list[list[float]]:
        seen_prefix.append(prefix)
        return [[0.0] * embedder.dim for _ in texts]

    embedder._embed_one_batch = fake_batch  # type: ignore[method-assign]
    embedder.embed_query("hello")
    assert len(seen_prefix) == 1
    assert seen_prefix[0] == embedder._query_prefix


@pytest.mark.parametrize("bad_batch_size", [0, -1, -100])
def test_constructor_rejects_nonpositive_batch_size(
    bad_batch_size: int, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        NomicTextEmbedder(
            tmp_path / "fake.onnx",
            tmp_path / "fake.tok",
            batch_size=bad_batch_size,
        )
