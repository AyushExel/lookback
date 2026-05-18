"""Mock embedders for tests — deterministic, fast, no model downloads.

Both mocks derive a 64-bit seed from a Blake2b hash of the input (text bytes
for text, file bytes for images) and use that seed with NumPy's RNG to draw
a Gaussian vector that we L2-normalize. Identical inputs produce identical
vectors; that's the property the indexer + store tests rely on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from lookback.embed.base import ImageEmbedder, TextEmbedder


def _seed_from_bytes(data: bytes) -> int:
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _hash_to_unit_vector(data: bytes, dim: int) -> list[float]:
    rng = np.random.default_rng(_seed_from_bytes(data))
    v = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        v = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32)
    else:
        v /= norm
    return v.tolist()


class MockTextEmbedder(TextEmbedder):
    """Deterministic text embedder for tests."""

    def __init__(self, dim: int = 768) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def normalized(self) -> bool:
        return True

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_unit_vector(t.encode("utf-8"), self._dim) for t in texts]


class MockImageEmbedder(ImageEmbedder):
    """Deterministic image embedder for tests — hashes file bytes."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def normalized(self) -> bool:
        return True

    def embed_batch(self, image_paths: list[Path]) -> list[list[float]]:
        return [_hash_to_unit_vector(Path(p).read_bytes(), self._dim) for p in image_paths]
