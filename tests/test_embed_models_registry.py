"""Feature under test: the embedding model registry — every entry carries
the metadata the CLI and download pipeline expect, and lookup helpers raise
clear errors for unknown names.
"""

from __future__ import annotations

import pytest

from lookback.embed.models import (
    IMAGE_MODELS,
    TEXT_MODELS,
    image_model,
    text_model,
)


def test_text_registry_contains_v1_5_and_v2_moe() -> None:
    assert "nomic-v1.5" in TEXT_MODELS
    assert "nomic-v2-moe" in TEXT_MODELS


def test_image_registry_contains_mobileclip_s2() -> None:
    assert "mobileclip-s2" in IMAGE_MODELS


def test_text_model_lookup_returns_spec() -> None:
    spec = text_model("nomic-v1.5")
    assert spec.dim == 768
    assert spec.max_length == 512
    assert spec.document_prefix.endswith(": ")
    assert spec.query_prefix.endswith(": ")
    assert spec.hf_repo.startswith("nomic-ai/")
    assert spec.min_ram_gb > 0
    assert spec.approx_disk_mb > 0


def test_image_model_lookup_returns_spec() -> None:
    spec = image_model("mobileclip-s2")
    assert spec.dim == 512
    assert spec.image_size == 256
    assert len(spec.mean) == 3
    assert len(spec.std) == 3
    # We point at a community ONNX export of Apple's checkpoint; the spec
    # carries the HF repo path the download command will hit.
    assert "mobileclip" in spec.hf_repo.lower()
    assert spec.onnx_filename.endswith(".onnx")
    # The plhery export uses raw [0,1] pixels (no ImageNet normalization).
    assert spec.mean == (0.0, 0.0, 0.0)
    assert spec.std == (1.0, 1.0, 1.0)


def test_text_model_unknown_name_lists_valid_options() -> None:
    with pytest.raises(ValueError, match=r"nomic-v1\.5"):
        text_model("not-real")


def test_image_model_unknown_name_lists_valid_options() -> None:
    with pytest.raises(ValueError, match="mobileclip-s2"):
        image_model("not-real")
