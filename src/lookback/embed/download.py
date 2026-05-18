"""Model-weight download helpers backed by ``huggingface_hub``.

We deliberately use per-file ``hf_hub_download`` calls instead of
``snapshot_download`` so the local layout exactly mirrors each spec's
``onnx_filename`` / ``tokenizer_filename`` strings — the CLI dispatch in
``cli.py`` builds adapter paths without any path-mangling.

When an ``ImageModelSpec`` carries ``text_onnx_filename`` (the joint-space
text encoder for cross-modal search), we download it alongside the vision
encoder and tokenizer in the same call so a single
``lookback models download mobileclip-s2`` gives the user everything they
need for both image embedding and text->image search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lookback.embed.models import ImageModelSpec, TextModelSpec


@dataclass(frozen=True, slots=True)
class TextModelPaths:
    model: Path
    tokenizer: Path


@dataclass(frozen=True, slots=True)
class ImageModelPaths:
    vision: Path
    text: Path | None
    tokenizer: Path | None


def download_text_model(spec: TextModelSpec, target_dir: Path) -> TextModelPaths:
    from huggingface_hub import hf_hub_download

    target_dir = Path(target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=spec.hf_repo,
        filename=spec.onnx_filename,
        local_dir=str(target_dir),
    )
    tokenizer_path = hf_hub_download(
        repo_id=spec.hf_repo,
        filename=spec.tokenizer_filename,
        local_dir=str(target_dir),
    )
    return TextModelPaths(model=Path(model_path), tokenizer=Path(tokenizer_path))


def download_image_model(spec: ImageModelSpec, target_dir: Path) -> ImageModelPaths:
    """Download the vision encoder + (when defined) the joint text encoder
    and its tokenizer. All three files end up under ``target_dir``.
    """
    from huggingface_hub import hf_hub_download

    target_dir = Path(target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    vision = Path(
        hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.onnx_filename,
            local_dir=str(target_dir),
        )
    )

    text: Path | None = None
    if spec.text_onnx_filename:
        text = Path(
            hf_hub_download(
                repo_id=spec.hf_repo,
                filename=spec.text_onnx_filename,
                local_dir=str(target_dir),
            )
        )

    tokenizer: Path | None = None
    if spec.tokenizer_filename:
        tokenizer = Path(
            hf_hub_download(
                repo_id=spec.hf_repo,
                filename=spec.tokenizer_filename,
                local_dir=str(target_dir),
            )
        )

    return ImageModelPaths(vision=vision, text=text, tokenizer=tokenizer)


def text_model_target_dir(models_root: Path, name: str) -> Path:
    return Path(models_root).expanduser() / name


def image_model_target_dir(models_root: Path, name: str) -> Path:
    return Path(models_root).expanduser() / name
