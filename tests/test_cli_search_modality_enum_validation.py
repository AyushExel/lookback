"""Feature under test: ``--modality`` is an enum-typed flag so typer
validates the value at parse time and surfaces the allowed set in
``--help``. Bad values exit cleanly with a clear message before any
real work happens (no embedder load, no Lance open).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app

runner = CliRunner()


def _cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    return cfg


def test_help_lists_modality_choices(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    # Typer formats Enum choices as ``[text|image|all]`` in the help text.
    assert "[text|image|all]" in result.output


def test_invalid_modality_exits_with_helpful_error(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    result = runner.invoke(
        app, ["search", "x", "--config", str(cfg), "--modality", "nope"]
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    # Typer's BadParameter mentions the offending value and lists choices.
    assert "modality" in combined.lower()


def test_help_lists_transport_choices_on_serve() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "[stdio|http]" in result.output
