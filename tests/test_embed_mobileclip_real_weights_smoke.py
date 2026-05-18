"""Feature under test: the real ``MobileCLIPImageEmbedder`` pipeline — when
the ``mobileclip-s2`` ONNX vision encoder is downloaded to the default
location, the adapter loads it, embeds real images, returns L2-normalized
512-d vectors, behaves deterministically across calls, and produces
*semantically meaningful* embeddings (visually similar images are closer
than dissimilar ones).

Auto-skips when the weights are absent. Mirrors the Nomic smoke test —
the goal is to catch regressions in the ONNX input feed
(``pixel_values`` name + layout), preprocessing (raw [0,1] pixels, no
ImageNet normalization on this specific export), and L2-normalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.needs_models

MODEL_DIR = Path.home() / ".lookback" / "models" / "mobileclip-s2"
MODEL_PATH = MODEL_DIR / "onnx" / "s2" / "vision_model.onnx"


def _skip_if_weights_missing() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(
            f"MobileCLIP-S2 weights not present at {MODEL_PATH}; "
            "run `lookback models download mobileclip-s2` to enable this test."
        )


def _solid_color_png(path: Path, rgb: tuple[int, int, int]) -> Path:
    """Generate a 320x240 solid-color PNG. Small enough for the test to be
    cheap; large enough that the resize-and-crop preprocessing actually does
    work (so we exercise the real pipeline, not a degenerate case).
    """
    Image.new("RGB", (320, 240), color=rgb).save(path)
    return path


def test_real_mobileclip_returns_l2_normalized_512d_vector(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    from lookback.embed.mobileclip import MobileCLIPImageEmbedder

    img = _solid_color_png(tmp_path / "red.png", (220, 30, 30))
    embedder = MobileCLIPImageEmbedder(MODEL_PATH)
    vec = embedder.embed_one(img)
    assert len(vec) == 512
    norm = float(np.linalg.norm(np.asarray(vec)))
    assert abs(norm - 1.0) < 1e-4, f"expected L2 norm ~1.0, got {norm}"


def test_real_mobileclip_is_deterministic_across_calls(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    from lookback.embed.mobileclip import MobileCLIPImageEmbedder

    img = _solid_color_png(tmp_path / "green.png", (30, 220, 30))
    embedder = MobileCLIPImageEmbedder(MODEL_PATH)
    a = embedder.embed_one(img)
    b = embedder.embed_one(img)
    assert a == b


def test_real_mobileclip_clusters_visually_similar_images(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    from lookback.embed.mobileclip import MobileCLIPImageEmbedder

    red_a = _solid_color_png(tmp_path / "red_a.png", (220, 30, 30))
    red_b = _solid_color_png(tmp_path / "red_b.png", (240, 60, 60))
    blue = _solid_color_png(tmp_path / "blue.png", (30, 30, 220))

    embedder = MobileCLIPImageEmbedder(MODEL_PATH)
    va, vb, vc = embedder.embed_batch([red_a, red_b, blue])
    va, vb, vc = np.asarray(va), np.asarray(vb), np.asarray(vc)

    # All L2-normalized, so dot product == cosine similarity.
    sim_similar = float(va @ vb)
    sim_dissimilar = float(va @ vc)
    assert sim_similar > sim_dissimilar, (
        f"two reds should be closer than red vs blue; "
        f"sim_similar={sim_similar:.4f}, sim_dissimilar={sim_dissimilar:.4f}"
    )
