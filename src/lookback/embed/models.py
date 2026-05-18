"""Embedding model registry — specs for text and image embedders.

Each spec carries every piece of information needed to (a) recommend the
model to a user, (b) download the right files from Hugging Face, and (c)
construct the runtime adapter pointing at those files. The CLI's
``init`` and ``models`` subcommands read this registry; runtime dispatch
in ``cli.py`` looks up the same specs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextModelSpec:
    name: str  # stable identifier used in config.toml
    display_name: str
    hf_repo: str
    onnx_filename: str  # relative path within the repo
    tokenizer_filename: str
    dim: int
    max_length: int
    document_prefix: str
    query_prefix: str
    approx_disk_mb: int
    min_ram_gb: float
    description: str


@dataclass(frozen=True, slots=True)
class ImageModelSpec:
    name: str
    display_name: str
    hf_repo: str
    onnx_filename: str  # vision encoder
    dim: int  # output dim of *both* vision and (when present) text encoders
    image_size: int  # square crop side
    mean: tuple[float, float, float]  # ImageNet-style normalization
    std: tuple[float, float, float]
    pixel_input_name: str  # ONNX input name (vision)
    approx_disk_mb: int
    min_ram_gb: float
    description: str
    # Optional joint-space text encoder for cross-modal text->image search.
    # When present, the download command also fetches the text ONNX + tokenizer
    # so ``lookback search`` can query the image table by free-text.
    text_onnx_filename: str | None = None
    tokenizer_filename: str | None = None
    text_max_length: int = 77  # CLIP standard
    text_input_name: str = "input_ids"  # ONNX input name (text)


TEXT_MODELS: dict[str, TextModelSpec] = {
    "nomic-v1.5": TextModelSpec(
        name="nomic-v1.5",
        display_name="Nomic Embed v1.5 (137M, dense, Matryoshka)",
        hf_repo="nomic-ai/nomic-embed-text-v1.5",
        onnx_filename="onnx/model.onnx",
        tokenizer_filename="tokenizer.json",
        dim=768,
        max_length=512,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        approx_disk_mb=550,
        min_ram_gb=4.0,
        description=(
            "Dense 137M-param model, 768-d output with Matryoshka truncation. "
            "Best quality-to-size on CPU and the safe default for laptops."
        ),
    ),
    "nomic-v2-moe": TextModelSpec(
        name="nomic-v2-moe",
        display_name="Nomic Embed v2 MoE (475M total / ~150M active)",
        hf_repo="nomic-ai/nomic-embed-text-v2-moe",
        onnx_filename="onnx/model.onnx",
        tokenizer_filename="tokenizer.json",
        dim=768,
        max_length=512,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        approx_disk_mb=1900,
        min_ram_gb=12.0,
        description=(
            "Mixture-of-experts model: ~150M active params per token but the full "
            "475M must be resident in RAM. Higher quality on retrieval, heavier "
            "footprint — pick only if you have ≥16 GB RAM and want the quality bump."
        ),
    ),
}


IMAGE_MODELS: dict[str, ImageModelSpec] = {
    "mobileclip-s2": ImageModelSpec(
        name="mobileclip-s2",
        display_name="MobileCLIP2-S2 (~150M, 512-d)",
        # Apple ships PyTorch checkpoints via CDN only; this community export
        # publishes the vision/text ONNX pair for transformers.js. The vision
        # tower is what we need.
        hf_repo="plhery/mobileclip2-onnx",
        onnx_filename="onnx/s2/vision_model.onnx",
        dim=512,
        image_size=256,
        # MobileCLIP2's ONNX exports apply NO normalization: raw [0, 1] pixels.
        # Don't paste the ImageNet defaults here even though it's a CLIP-like
        # model — that would silently degrade embedding quality.
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        pixel_input_name="pixel_values",
        approx_disk_mb=220,
        min_ram_gb=4.0,
        description=(
            "Apple's MobileCLIP2-S2: matches SigLIP-SO400M at 2x fewer params, "
            "3-15 ms inference on Apple Silicon. The v0 image-embedder choice."
        ),
        text_onnx_filename="onnx/s2/text_model.onnx",
        tokenizer_filename="tokenizer.json",
        text_max_length=77,
        text_input_name="input_ids",
    ),
}


def text_model(name: str) -> TextModelSpec:
    try:
        return TEXT_MODELS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(TEXT_MODELS))
        raise ValueError(f"unknown text model {name!r}; expected one of: {valid}") from exc


def image_model(name: str) -> ImageModelSpec:
    try:
        return IMAGE_MODELS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(IMAGE_MODELS))
        raise ValueError(f"unknown image model {name!r}; expected one of: {valid}") from exc
