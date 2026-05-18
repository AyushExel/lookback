"""FastMCP server exposing Lookback as Model Context Protocol tools.

External AI tools — Claude Code, Cursor, Continue, ChatGPT Desktop, etc. —
connect via the MCP protocol and call ``lookback_search`` to retrieve
context from the user's local index. The server runs on stdio by default
(every modern MCP-capable client supports stdio), with optional HTTP
transport for hosted scenarios.

See ``MCP_SETUP.md`` for client integration recipes.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lookback.config import LookbackConfig
from lookback.mcp.tools import ToolContext, search, stats


def create_server(config: LookbackConfig, *, name: str = "lookback") -> FastMCP:
    """Build a FastMCP server with Lookback's tools registered.

    Returns an unstarted server — the caller picks the transport:
    - ``server.run()`` → stdio (the path IDE integrations use)
    - ``server.run(transport="http", port=...)`` → HTTP
    - The in-process ``fastmcp.Client`` for tests
    """
    mcp = FastMCP(name)
    ctx = ToolContext(config=config)

    @mcp.tool
    def lookback_search(
        query: str,
        modality: str = "all",
        limit: int = 10,
        source_kind: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Semantic search across the user's locally indexed files.

        Returns a dict with two arrays:
        - ``"text"`` — markdown / code / PDF chunk hits ranked by Nomic.
        - ``"image"`` — screenshot hits ranked by joint-space text->image
          similarity (MobileCLIP). Empty unless the user indexed images.

        Args:
            query: Free-text query.
            modality: ``"text"``, ``"image"``, or ``"all"`` (default).
            limit: Max results per modality, 1-100.
            source_kind: Optional filter (``"markdown"``, ``"python"``,
                ``"screenshot"``, ``"pdf"``, ``"plaintext"``).
        """
        return search(
            ctx,
            query=query,
            modality=modality,
            limit=limit,
            source_kind=source_kind,
        )

    @mcp.tool
    def lookback_stats() -> dict[str, int]:
        """Return row counts per table in the local index.

        Useful for telling the user whether their index is populated and
        debugging "why am I not getting hits" cases.
        """
        return stats(ctx)

    return mcp
