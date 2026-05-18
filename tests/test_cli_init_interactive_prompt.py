"""Feature under test: ``lookback init --interactive`` prompts for the text
and image embedder choice, accepting the recommended default on a blank
input. We feed empty lines via the CLI runner's ``input`` parameter and
assert the config still lands valid model names.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app
from lookback.config import LookbackConfig

runner = CliRunner()


def test_interactive_init_accepts_defaults_on_empty_input(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    # Two empty lines = accept the default for both prompts.
    result = runner.invoke(app, ["init", "--config", str(cfg), "--interactive"], input="\n\n")
    assert result.exit_code == 0, result.output
    loaded = LookbackConfig.load(cfg)
    assert loaded.text_embedder in {"mock", "nomic-v1.5", "nomic-v2-moe"}
    assert loaded.image_embedder in {"mock", "mobileclip-s2"}


def test_interactive_init_accepts_explicit_text_choice(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    # First prompt: 'nomic-v2-moe'. Second prompt: accept default.
    result = runner.invoke(
        app,
        ["init", "--config", str(cfg), "--interactive"],
        input="nomic-v2-moe\n\n",
    )
    assert result.exit_code == 0, result.output
    loaded = LookbackConfig.load(cfg)
    assert loaded.text_embedder == "nomic-v2-moe"
