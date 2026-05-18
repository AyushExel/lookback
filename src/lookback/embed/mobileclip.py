"""MobileCLIP2-S2 image adapter — ONNX Runtime + PIL preprocessing, lazy-loaded.

Mirrors the lazy-load contract of :mod:`lookback.embed.nomic`: the
constructor touches no files, and the ONNX session materializes only on
the first ``embed_batch`` call. File-existence checks fire before the
``onnxruntime`` import so missing weights surface as ``FileNotFoundError``
even on a clean install without the ``[image]`` extras.

Image preprocessing follows the standard OpenCLIP / MobileCLIP2 pipeline:
resize the shorter side to ``image_size``, center-crop to ``image_size``
square, scale to [0, 1], and normalize with ImageNet mean/std. The
post-pool output is L2-normalized so search can use the perf-guide's
recommended ``dot`` distance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lookback.embed.base import ImageEmbedder
from lookback.embed.models import IMAGE_MODELS, ImageModelSpec, image_model

logger = logging.getLogger(__name__)


class MobileCLIPImageEmbedder(ImageEmbedder):
    def __init__(
        self,
        model_path: Path | str,
        *,
        spec: ImageModelSpec | None = None,
        model_name: str = "mobileclip-s2",
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._spec = spec if spec is not None else image_model(model_name)
        self._session: object | None = None

    @property
    def dim(self) -> int:
        return self._spec.dim

    @property
    def normalized(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"MobileCLIP model not found at {self._model_path}. Download with: "
                f"lookback models download {self._spec.name}"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - extras-missing path
            raise ImportError(
                "MobileCLIPImageEmbedder requires onnxruntime; install via "
                "`uv pip install lookback[image]`."
            ) from exc
        logger.info("loading MobileCLIP ONNX session: %s", self._model_path)
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )

    def embed_batch(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []
        self._ensure_loaded()
        assert self._session is not None

        batch = np.stack(
            [self._preprocess(Path(p)) for p in image_paths], axis=0
        ).astype(np.float32)
        outputs = self._session.run(None, {self._spec.pixel_input_name: batch})
        embeddings = np.asarray(outputs[0], dtype=np.float32)

        if embeddings.shape[-1] != self._spec.dim:
            embeddings = embeddings[..., : self._spec.dim]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        return [vec.tolist() for vec in normalized]

    def _preprocess(self, image_path: Path) -> np.ndarray:
        from PIL import Image

        side = self._spec.image_size
        img = Image.open(image_path).convert("RGB")

        # Resize so the shorter side matches `side`, then center-crop a square.
        w, h = img.size
        scale = side / min(w, h)
        new_size = (max(side, round(w * scale)), max(side, round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        left = (img.size[0] - side) // 2
        top = (img.size[1] - side) // 2
        img = img.crop((left, top, left + side, top + side))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        mean = np.asarray(self._spec.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self._spec.std, dtype=np.float32).reshape(3, 1, 1)
        return (arr - mean) / std


class MobileCLIPTextEmbedder:
    """Encodes text in MobileCLIP's joint vision-language embedding space.

    Returns ``dim``-d L2-normalized vectors that live in the *same* space as
    ``MobileCLIPImageEmbedder`` outputs, so ``LanceStore.search_image`` can
    be queried directly with a vector produced here. This is what makes
    text->image cross-modal search possible.

    Deliberately not a :class:`~lookback.embed.base.TextEmbedder` subclass:
    its output dim (512 for S2) doesn't match ``TEXT_EMBED_DIM`` (768), and
    the indexer's dim-validation would (correctly) reject it. Conceptually
    it's an image-table query encoder, not a text-table document encoder.
    """

    def __init__(
        self,
        model_path: Path | str,
        tokenizer_path: Path | str,
        *,
        spec: ImageModelSpec | None = None,
        model_name: str = "mobileclip-s2",
    ) -> None:
        self._model_path = Path(model_path).expanduser()
        self._tokenizer_path = Path(tokenizer_path).expanduser()
        self._spec = spec if spec is not None else image_model(model_name)
        self._session: object | None = None
        self._tokenizer: object | None = None

    @property
    def dim(self) -> int:
        return self._spec.dim

    @property
    def normalized(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        # File-existence first, matching the rest of the embedder layer.
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"MobileCLIP text encoder not found at {self._model_path}. "
                f"Run: lookback models download {self._spec.name}"
            )
        if not self._tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {self._tokenizer_path}. "
                f"Run: lookback models download {self._spec.name}"
            )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MobileCLIPTextEmbedder requires onnxruntime and tokenizers."
            ) from exc
        logger.info("loading MobileCLIP text ONNX session: %s", self._model_path)
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        tok = Tokenizer.from_file(str(self._tokenizer_path))
        tok.enable_truncation(max_length=self._spec.text_max_length)
        tok.enable_padding(length=self._spec.text_max_length)
        self._tokenizer = tok

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._session is not None
        assert self._tokenizer is not None

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)

        # Some CLIP text-encoder ONNX exports require only input_ids; others
        # also accept attention_mask. Introspect and pass what the model wants.
        input_names = {inp.name for inp in self._session.get_inputs()}
        feed: dict[str, np.ndarray] = {self._spec.text_input_name: input_ids}
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.asarray(
                [e.attention_mask for e in encodings], dtype=np.int64
            )

        outputs = self._session.run(None, feed)
        embeddings = np.asarray(outputs[0], dtype=np.float32)
        if embeddings.shape[-1] != self._spec.dim:
            embeddings = embeddings[..., : self._spec.dim]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return [(vec / n).tolist() for vec, n in zip(embeddings, norms, strict=True)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


__all__ = [
    "IMAGE_MODELS",
    "MobileCLIPImageEmbedder",
    "MobileCLIPTextEmbedder",
]
