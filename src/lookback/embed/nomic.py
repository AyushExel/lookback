"""Nomic Embed text adapter — ONNX Runtime + HF tokenizers, lazy-loaded.

Weights are not bundled. Users download them out-of-band (e.g. via
``huggingface-cli download nomic-ai/nomic-embed-text-v1.5 --local-dir
~/.lookback/models/nomic``) and point the adapter at the model file. The
adapter loads nothing at construction; the model and tokenizer only land in
memory on the first ``embed_batch`` call. This keeps CLI startup fast and
the test path mock-only by default.

Nomic Embed v1.5 (137M, Matryoshka) is the v0 default — small enough for
CPU inference, top-tier quality-to-size ratio per our 2026 research notes.
Architecture-wise the adapter is dim/model-agnostic; pass a different ONNX
file (e.g. v2-MoE) and an updated ``dim``/``max_length`` and it works.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lookback.embed.base import TextEmbedder

logger = logging.getLogger(__name__)

DEFAULT_DIM = 768
DEFAULT_MAX_LENGTH = 512
DEFAULT_DOCUMENT_PREFIX = "search_document: "
DEFAULT_QUERY_PREFIX = "search_query: "
# Cap on how many sequences we hand to ONNX in a single call. For Nomic
# v1.5 with seq_len=512 and hidden=768, each batch of 32 produces a
# (32, 512, 768) float32 last_hidden_state tensor — ~48 MB intermediate.
# Without this cap, embedding a single large file (a PDF or log producing
# thousands of chunks) explodes RAM into the multi-GB range. See
# https://github.com/AyushExel/lookback#performance for context.
DEFAULT_BATCH_SIZE = 32


class NomicTextEmbedder(TextEmbedder):
    def __init__(
        self,
        model_path: Path | str,
        tokenizer_path: Path | str | None = None,
        *,
        dim: int = DEFAULT_DIM,
        max_length: int = DEFAULT_MAX_LENGTH,
        document_prefix: str = DEFAULT_DOCUMENT_PREFIX,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._model_path = Path(model_path).expanduser()
        if tokenizer_path is None:
            self._tokenizer_path = self._model_path.parent / "tokenizer.json"
        else:
            self._tokenizer_path = Path(tokenizer_path).expanduser()
        self._dim = dim
        self._max_length = max_length
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._session: object | None = None
        self._tokenizer: object | None = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def normalized(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        # Check file existence first so the error is configuration-level,
        # independent of whether the optional [text] extras are installed.
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Nomic model not found at {self._model_path}. Download with: "
                "huggingface-cli download nomic-ai/nomic-embed-text-v1.5 "
                f"--local-dir {self._model_path.parent}"
            )
        if not self._tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {self._tokenizer_path}. Expected alongside the model."
            )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only when extras missing
            raise ImportError(
                "NomicTextEmbedder requires onnxruntime and tokenizers; install via "
                "`uv pip install lookback[text]` or `pip install lookback[text]`."
            ) from exc
        logger.info("loading Nomic ONNX session: %s", self._model_path)
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        tok = Tokenizer.from_file(str(self._tokenizer_path))
        tok.enable_truncation(max_length=self._max_length)
        tok.enable_padding(length=self._max_length)
        self._tokenizer = tok

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, prefix=self._document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], prefix=self._query_prefix)[0]

    def _embed(self, texts: list[str], *, prefix: str) -> list[list[float]]:
        """Mini-batched ONNX inference.

        Splits ``texts`` into chunks of ``self._batch_size`` and runs the
        session once per chunk, so peak memory is bounded by the per-batch
        intermediate tensor (~48 MB for the defaults) regardless of how
        many chunks a single file produces.
        """
        if not texts:
            return []
        self._ensure_loaded()
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            sub = texts[start : start + self._batch_size]
            out.extend(self._embed_one_batch(sub, prefix=prefix))
        return out

    def _embed_one_batch(
        self, texts: list[str], *, prefix: str
    ) -> list[list[float]]:
        assert self._session is not None
        assert self._tokenizer is not None

        prefixed = [prefix + t for t in texts]
        encodings = self._tokenizer.encode_batch(prefixed)
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

        # Nomic's ONNX export inherits BERT's three-input contract. token_type_ids
        # is all zeros for single-sequence inputs — the embedder never sees pairs.
        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feed)
        # Convention: outputs[0] is last_hidden_state (batch, seq, hidden).
        last_hidden = np.asarray(outputs[0], dtype=np.float32)
        mask = attention_mask[..., None].astype(np.float32)
        denom = mask.sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        pooled = (last_hidden * mask).sum(axis=1) / denom

        if pooled.shape[-1] != self._dim:
            # Matryoshka-style truncation if the model is bigger than configured.
            pooled = pooled[..., : self._dim]

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = pooled / norms
        return [vec.astype(np.float32).tolist() for vec in normalized]
