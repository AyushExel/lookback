"""Feature under test: ``lookback serve`` validates its CLI surface — the
``--transport`` flag accepts only ``stdio`` and ``http``, and unknown
values are rejected before any server boots.

We deliberately don't try to actually start the MCP server in the test
suite (that would block on stdio or bind a port). The ``create_server``
function is covered by ``test_mcp_server_registration.py``.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app

runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    return cfg


def test_unknown_transport_rejected(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app, ["serve", "--config", str(cfg), "--transport", "websocket"]
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "transport" in combined.lower()


def test_serve_command_is_registered() -> None:
    """``lookback serve --help`` must enumerate the transport flag — the
    help text is what IDE config docs link to."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "stdio" in result.output
    assert "http" in result.output
    assert "--transport" in result.output
