"""Embedding layer — abstract bases plus mock and real adapters."""

from lookback.embed.base import ImageEmbedder, TextEmbedder
from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder

__all__ = [
    "ImageEmbedder",
    "MockImageEmbedder",
    "MockTextEmbedder",
    "TextEmbedder",
]
