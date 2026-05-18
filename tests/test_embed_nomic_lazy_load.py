"""Feature under test: ``NomicTextEmbedder`` does not load any model files at
construction time. The session and tokenizer materialize only on the first
``embed_batch`` / ``embed_query`` call.

This is the property that keeps CLI startup snappy when the embedder is
configured but no work needs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.embed.nomic import NomicTextEmbedder


def test_constructor_does_not_touch_disk(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-model.onnx"
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    # Neither file exists; constructor must not raise.
    embedder = NomicTextEmbedder(bogus_model, bogus_tok)
    assert embedder.dim == 768
    assert embedder.normalized is True


def test_missing_model_file_raises_clear_error_on_first_use(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-model.onnx"
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    embedder = NomicTextEmbedder(bogus_model, bogus_tok)
    with pytest.raises(FileNotFoundError, match=str(bogus_model)):
        embedder.embed_batch(["anything"])


def test_missing_tokenizer_raises_clear_error_on_first_use(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")  # exists but bogus content; we never reach load
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    embedder = NomicTextEmbedder(model, bogus_tok)
    with pytest.raises(FileNotFoundError, match=str(bogus_tok)):
        embedder.embed_batch(["anything"])
