"""Plain-Python tool implementations behind the MCP server.

Each function takes a ``LookbackConfig`` and returns JSON-serializable
output. The MCP server (``lookback.mcp.server``) wraps these with
``@mcp.tool`` so the same logic is testable both directly and via the
MCP protocol layer.

Embedders are loaded lazily via a small cache so an idle MCP server
doesn't pay the Nomic / MobileCLIP cold-start cost until a tool is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lookback.config import LookbackConfig
from lookback.embed.factory import (
    build_image_text_embedder,
    build_text_embedder,
)
from lookback.store.lance_store import LanceStore

VALID_MODALITIES = frozenset({"text", "image", "all"})


@dataclass
class ToolContext:
    """Per-server lazy-loaded resources, scoped to one ``create_server`` call."""

    config: LookbackConfig
    _store: LanceStore | None = None
    _text_embedder: object | None = None
    _image_text_embedder: object | None = None

    def store(self) -> LanceStore:
        if self._store is None:
            self._store = LanceStore(self.config.data_dir)
        return self._store

    def text_embedder(self):
        if self._text_embedder is None:
            self._text_embedder = build_text_embedder(self.config)
        return self._text_embedder

    def image_text_embedder(self):
        if self._image_text_embedder is None:
            self._image_text_embedder = build_image_text_embedder(self.config)
        return self._image_text_embedder


def search(
    ctx: ToolContext,
    query: str,
    *,
    modality: str = "all",
    limit: int = 10,
    source_kind: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run text and/or image-table search and return results keyed by modality."""
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"modality must be one of {sorted(VALID_MODALITIES)}, got {modality!r}"
        )
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    where = f"source_kind = '{source_kind}'" if source_kind else None
    store = ctx.store()

    text_hits: list[dict[str, Any]] = []
    image_hits: list[dict[str, Any]] = []

    if modality in {"text", "all"}:
        qv = ctx.text_embedder().embed_query(query)
        text_hits = store.search_text(qv, limit=limit, where=where)

    if modality in {"image", "all"}:
        qv = ctx.image_text_embedder().embed_query(query)
        image_hits = store.search_image(qv, limit=limit, where=where)

    return {"text": text_hits, "image": image_hits}


def stats(ctx: ToolContext) -> dict[str, int]:
    """Return row counts for every table in the index."""
    return ctx.store().stats()
