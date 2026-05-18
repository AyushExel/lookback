"""Feature under test: ``MobileCLIPImageEmbedder`` does not load any model
files at construction, and a missing weight file surfaces as a clear
``FileNotFoundError`` on first use — independent of whether the optional
``[image]`` extras are installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.embed.mobileclip import MobileCLIPImageEmbedder


def test_constructor_does_not_touch_disk(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-model.onnx"
    embedder = MobileCLIPImageEmbedder(bogus_model)
    assert embedder.dim == 512
    assert embedder.normalized is True


def test_missing_model_raises_file_not_found_on_first_use(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-model.onnx"
    fake_image = tmp_path / "shot.png"
    fake_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    embedder = MobileCLIPImageEmbedder(bogus_model)
    with pytest.raises(FileNotFoundError, match=str(bogus_model)):
        embedder.embed_batch([fake_image])


def test_empty_input_returns_empty_without_loading(tmp_path: Path) -> None:
    bogus_model = tmp_path / "no-such-model.onnx"
    embedder = MobileCLIPImageEmbedder(bogus_model)
    # Empty input must short-circuit before any load is attempted.
    assert embedder.embed_batch([]) == []
