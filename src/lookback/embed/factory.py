"""Factory functions that turn a ``LookbackConfig`` into concrete embedders.

These live here (and not in ``lookback.cli``) so the MCP server can reuse
the exact same dispatch without importing the Typer app. Every place that
needs to build an embedder for a given config goes through these helpers.
"""

from __future__ import annotations

from lookback.config import LookbackConfig
from lookback.embed.base import ImageEmbedder, TextEmbedder
from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.embed.models import IMAGE_MODELS, TEXT_MODELS, image_model, text_model
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM


def build_text_embedder(config: LookbackConfig) -> TextEmbedder:
    name = config.text_embedder.lower()
    if name == "mock":
        return MockTextEmbedder(dim=TEXT_EMBED_DIM)
    if name in TEXT_MODELS:
        from lookback.embed.nomic import NomicTextEmbedder

        spec = text_model(name)
        model_dir = config.models_dir / name
        return NomicTextEmbedder(
            model_dir / spec.onnx_filename,
            model_dir / spec.tokenizer_filename,
            dim=spec.dim,
            max_length=spec.max_length,
            document_prefix=spec.document_prefix,
            query_prefix=spec.query_prefix,
        )
    valid = ["mock", *sorted(TEXT_MODELS.keys())]
    raise ValueError(
        f"unknown text_embedder: {config.text_embedder!r} (expected one of {valid})"
    )


def build_image_embedder(config: LookbackConfig) -> ImageEmbedder:
    name = config.image_embedder.lower()
    if name == "mock":
        return MockImageEmbedder(dim=IMAGE_EMBED_DIM)
    if name in IMAGE_MODELS:
        from lookback.embed.mobileclip import MobileCLIPImageEmbedder

        spec = image_model(name)
        model_dir = config.models_dir / name
        return MobileCLIPImageEmbedder(
            model_dir / spec.onnx_filename,
            spec=spec,
        )
    valid = ["mock", *sorted(IMAGE_MODELS.keys())]
    raise ValueError(
        f"unknown image_embedder: {config.image_embedder!r} (expected one of {valid})"
    )


def build_image_text_embedder(config: LookbackConfig):
    """Build the encoder that produces vectors in the image embedder's joint
    space — used for text->image cross-modal search.

    Returns a ``MockTextEmbedder`` at 512-d when the config says
    ``image_embedder = "mock"`` so test plumbing works without real weights.
    """
    name = config.image_embedder.lower()
    if name == "mock":
        return MockTextEmbedder(dim=IMAGE_EMBED_DIM)
    if name in IMAGE_MODELS:
        from lookback.embed.mobileclip import MobileCLIPTextEmbedder

        spec = image_model(name)
        if not spec.text_onnx_filename or not spec.tokenizer_filename:
            raise ValueError(
                f"image model {name!r} has no joint-space text encoder; "
                "cross-modal text->image search isn't supported for this model"
            )
        model_dir = config.models_dir / name
        return MobileCLIPTextEmbedder(
            model_dir / spec.text_onnx_filename,
            model_dir / spec.tokenizer_filename,
            spec=spec,
        )
    valid = ["mock", *sorted(IMAGE_MODELS.keys())]
    raise ValueError(
        f"unknown image_embedder: {config.image_embedder!r} (expected one of {valid})"
    )
