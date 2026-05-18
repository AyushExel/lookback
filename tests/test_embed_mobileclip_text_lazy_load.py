"""Feature under test: ``MobileCLIPTextEmbedder`` matches the lazy-load
contract of its image counterpart — the constructor touches no disk, and
missing weight or tokenizer files surface as ``FileNotFoundError`` on the
first call. File-existence checks fire before the ``onnxruntime`` /
``tokenizers`` imports so configuration errors never get masked by a
missing-extras ImportError.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.embed.mobileclip import MobileCLIPTextEmbedder


def test_constructor_does_not_touch_disk(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-text.onnx"
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    embedder = MobileCLIPTextEmbedder(bogus_model, bogus_tok)
    assert embedder.dim == 512
    assert embedder.normalized is True


def test_missing_text_model_raises_file_not_found_on_first_use(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-text.onnx"
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    embedder = MobileCLIPTextEmbedder(bogus_model, bogus_tok)
    with pytest.raises(FileNotFoundError, match=str(bogus_model)):
        embedder.embed_batch(["anything"])


def test_missing_tokenizer_raises_file_not_found_on_first_use(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")  # exists but bogus; we never reach load
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    embedder = MobileCLIPTextEmbedder(model, bogus_tok)
    with pytest.raises(FileNotFoundError, match=str(bogus_tok)):
        embedder.embed_batch(["anything"])


def test_empty_input_short_circuits_without_loading(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-text.onnx"
    bogus_tok = tmp_path / "no-such-tokenizer.json"
    embedder = MobileCLIPTextEmbedder(bogus_model, bogus_tok)
    assert embedder.embed_batch([]) == []
