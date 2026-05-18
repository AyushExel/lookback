"""Feature under test: ``create_server`` returns a FastMCP instance with
both Lookback tools registered (``lookback_search`` and ``lookback_stats``)
and discoverable via the MCP server's tool-listing API.

We don't actually run the server in this test — that's a process-level
concern. We verify the registration surface: an MCP-capable client
introspecting our server will see the right tools with the right
parameter shapes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from lookback.config import LookbackConfig
from lookback.mcp.server import create_server


def _config(tmp_path: Path) -> LookbackConfig:
    return LookbackConfig(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        text_embedder="mock",
        image_embedder="mock",
    )


def test_create_server_returns_fastmcp_instance(tmp_path: Path) -> None:
    server = create_server(_config(tmp_path))
    # FastMCP servers expose `.name` regardless of version.
    assert server.name == "lookback"


def test_server_exposes_lookback_search_and_lookback_stats(tmp_path: Path) -> None:
    server = create_server(_config(tmp_path))
    # ``FastMCP.list_tools`` is async — resolve it from a fresh loop.
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    assert "lookback_search" in tool_names
    assert "lookback_stats" in tool_names


def test_lookback_search_tool_advertises_query_param(tmp_path: Path) -> None:
    server = create_server(_config(tmp_path))
    tools = asyncio.run(server.list_tools())
    search_tool = next(t for t in tools if t.name == "lookback_search")
    props = (search_tool.parameters or {}).get("properties", {})
    assert "query" in props, f"expected `query` parameter, got {list(props)}"
    assert "modality" in props
    assert "limit" in props
    # `query` must be required; others have defaults.
    required = set((search_tool.parameters or {}).get("required", []))
    assert "query" in required


def test_lookback_search_tool_carries_docstring_as_description(tmp_path: Path) -> None:
    server = create_server(_config(tmp_path))
    tools = asyncio.run(server.list_tools())
    search_tool = next(t for t in tools if t.name == "lookback_search")
    assert search_tool.description, "description must be non-empty for IDE UIs"
    assert "semantic search" in search_tool.description.lower()
