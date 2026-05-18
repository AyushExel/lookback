"""Feature under test: ``MobileCLIPTextEmbedder`` against the real
``mobileclip-s2`` weights — joint-space text embedding for cross-modal
text->image retrieval.

The semantic check is the load-bearing assertion: a text query about a
specific color must be more similar to an image of that color than to an
image of a different color. If this regresses, cross-modal search becomes
no better than random.

Auto-skips when weights aren't downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.needs_models

MODEL_DIR = Path.home() / ".lookback" / "models" / "mobileclip-s2"
VISION_PATH = MODEL_DIR / "onnx" / "s2" / "vision_model.onnx"
TEXT_PATH = MODEL_DIR / "onnx" / "s2" / "text_model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"


def _skip_if_weights_missing() -> None:
    if not (VISION_PATH.exists() and TEXT_PATH.exists() and TOKENIZER_PATH.exists()):
        pytest.skip(
            f"MobileCLIP-S2 joint weights not all present under {MODEL_DIR}; "
            "run `lookback models download mobileclip-s2` to enable this test."
        )


def _solid_color_png(path: Path, rgb: tuple[int, int, int]) -> Path:
    Image.new("RGB", (320, 240), color=rgb).save(path)
    return path


def test_real_text_encoder_returns_l2_normalized_512d_vector() -> None:
    _skip_if_weights_missing()
    from lookback.embed.mobileclip import MobileCLIPTextEmbedder

    embedder = MobileCLIPTextEmbedder(TEXT_PATH, TOKENIZER_PATH)
    v = embedder.embed_query("a friendly robot")
    assert len(v) == 512
    norm = float(np.linalg.norm(np.asarray(v)))
    assert abs(norm - 1.0) < 1e-4


def test_real_text_encoder_is_deterministic() -> None:
    _skip_if_weights_missing()
    from lookback.embed.mobileclip import MobileCLIPTextEmbedder

    embedder = MobileCLIPTextEmbedder(TEXT_PATH, TOKENIZER_PATH)
    a = embedder.embed_query("the same query string")
    b = embedder.embed_query("the same query string")
    assert a == b


def test_text_query_aligns_with_correctly_colored_image(tmp_path: Path) -> None:
    """The headline cross-modal property: a text query for one color must
    score the image of that color higher than the image of another color."""
    _skip_if_weights_missing()
    from lookback.embed.mobileclip import (
        MobileCLIPImageEmbedder,
        MobileCLIPTextEmbedder,
    )

    blue = _solid_color_png(tmp_path / "blue.png", (30, 30, 220))
    red = _solid_color_png(tmp_path / "red.png", (220, 30, 30))

    text_emb = MobileCLIPTextEmbedder(TEXT_PATH, TOKENIZER_PATH)
    img_emb = MobileCLIPImageEmbedder(VISION_PATH)

    img_blue = np.asarray(img_emb.embed_one(blue))
    img_red = np.asarray(img_emb.embed_one(red))

    q_blue = np.asarray(text_emb.embed_query("a blue square"))
    q_red = np.asarray(text_emb.embed_query("a red square"))

    sim_blue_to_blue = float(q_blue @ img_blue)
    sim_blue_to_red = float(q_blue @ img_red)
    sim_red_to_red = float(q_red @ img_red)
    sim_red_to_blue = float(q_red @ img_blue)

    assert sim_blue_to_blue > sim_blue_to_red, (
        f"'blue square' text should match blue image > red image; "
        f"got blue↔blue={sim_blue_to_blue:.4f}, blue↔red={sim_blue_to_red:.4f}"
    )
    assert sim_red_to_red > sim_red_to_blue, (
        f"'red square' text should match red image > blue image; "
        f"got red↔red={sim_red_to_red:.4f}, red↔blue={sim_red_to_blue:.4f}"
    )
