"""Feature under test: the underlying ``search`` tool function — the same
logic the MCP server exposes as ``lookback_search``. Tested directly
against a ``ToolContext`` so we exercise the dispatch without booting
the MCP protocol layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lookback.cli import app
from lookback.config import LookbackConfig
from lookback.mcp.tools import ToolContext, search


def _seed(tmp_path: Path) -> tuple[LookbackConfig, Path]:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.md").write_text("# Title\nSome text.\n")
    (src / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    runner = CliRunner()
    runner.invoke(app, ["index", str(src), "--config", str(cfg_path)])
    return LookbackConfig.load(cfg_path), src


def test_search_all_modality_returns_text_and_image_keys(tmp_path: Path) -> None:
    config, _ = _seed(tmp_path)
    ctx = ToolContext(config=config)
    out = search(ctx, "anything")
    assert set(out.keys()) == {"text", "image"}
    assert out["text"], "expected at least one text hit"
    assert out["image"], "expected at least one image hit"


def test_search_text_only_modality_skips_image_table(tmp_path: Path) -> None:
    config, _ = _seed(tmp_path)
    ctx = ToolContext(config=config)
    out = search(ctx, "anything", modality="text")
    assert out["text"]
    assert out["image"] == []


def test_search_image_only_modality_skips_text_table(tmp_path: Path) -> None:
    config, _ = _seed(tmp_path)
    ctx = ToolContext(config=config)
    out = search(ctx, "anything", modality="image")
    assert out["text"] == []
    assert out["image"]


def test_search_with_invalid_modality_raises(tmp_path: Path) -> None:
    config, _ = _seed(tmp_path)
    ctx = ToolContext(config=config)
    with pytest.raises(ValueError, match="modality"):
        search(ctx, "x", modality="bogus")


def test_search_with_invalid_limit_raises(tmp_path: Path) -> None:
    config, _ = _seed(tmp_path)
    ctx = ToolContext(config=config)
    with pytest.raises(ValueError, match="limit"):
        search(ctx, "x", limit=0)


def test_search_source_kind_filter_narrows_results(tmp_path: Path) -> None:
    config, _ = _seed(tmp_path)
    ctx = ToolContext(config=config)
    out = search(ctx, "anything", source_kind="markdown")
    for hit in out["text"]:
        assert hit["source_kind"] == "markdown"
