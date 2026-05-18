"""Abstract bases for text and image embedders.

Every embedder declares its output dimension and whether its vectors are
L2-normalized; that drives index choices downstream (``dot`` distance with
normalized vectors is the perf-guide-recommended path, ``cosine`` otherwise).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TextEmbedder(ABC):
    """A pure-text embedder: ``list[str] -> list[list[float]]``."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Length of every output vector."""

    @property
    @abstractmethod
    def normalized(self) -> bool:
        """Whether output vectors are L2-normalized."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings; output length must equal input length."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Adapters with task-specific prompting (e.g.
        Nomic's ``search_query:`` prefix) override this; the default reuses
        ``embed_batch``."""
        return self.embed_one(text)


class ImageEmbedder(ABC):
    """An image embedder: ``list[Path] -> list[list[float]]``."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def normalized(self) -> bool: ...

    @abstractmethod
    def embed_batch(self, image_paths: list[Path]) -> list[list[float]]: ...

    def embed_one(self, image_path: Path) -> list[float]:
        return self.embed_batch([image_path])[0]
