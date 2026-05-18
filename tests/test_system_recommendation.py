"""Feature under test: ``recommend_text_model`` and ``recommend_image_model``
return registered model names (so the CLI can immediately dispatch on the
recommendation without a second lookup).

The recommendation policy itself is deliberately simple in v0 — we don't
test the *content* of the recommendation here, only that it points at a
valid model.
"""

from __future__ import annotations

from lookback.embed.models import IMAGE_MODELS, TEXT_MODELS
from lookback.system import (
    SystemProfile,
    recommend_image_model,
    recommend_text_model,
)


def _profile(ram_gb: float | None) -> SystemProfile:
    return SystemProfile(
        os_name="Darwin",
        arch="arm64",
        is_apple_silicon=True,
        total_ram_gb=ram_gb,
        cpu_count=8,
    )


def test_recommended_text_model_is_in_registry_for_low_ram() -> None:
    assert recommend_text_model(_profile(4.0)) in TEXT_MODELS


def test_recommended_text_model_is_in_registry_for_high_ram() -> None:
    assert recommend_text_model(_profile(32.0)) in TEXT_MODELS


def test_recommended_text_model_is_in_registry_when_ram_unknown() -> None:
    assert recommend_text_model(_profile(None)) in TEXT_MODELS


def test_recommended_image_model_is_mobileclip_s2() -> None:
    assert recommend_image_model(_profile(8.0)) == "mobileclip-s2"
    assert recommend_image_model(_profile(8.0)) in IMAGE_MODELS
