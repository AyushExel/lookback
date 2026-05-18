"""Feature under test: ``lookback init`` writes a valid starter config file
to the location the user (or a test) specifies and refuses to overwrite an
existing file without ``--force``.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app
from lookback.config import LookbackConfig

runner = CliRunner()


def test_init_creates_a_loadable_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    result = runner.invoke(app, ["init", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert cfg.exists()
    # The file must be parseable by the same loader the CLI uses, and the
    # embedder values must be names the runtime dispatch recognises.
    loaded = LookbackConfig.load(cfg)
    assert loaded.text_embedder in {"mock", "nomic-v1.5", "nomic-v2-moe"}
    assert loaded.image_embedder in {"mock", "mobileclip-s2"}


def test_init_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("existing = true\n")
    result = runner.invoke(app, ["init", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "already exists" in result.output or "already exists" in (result.stderr or "")
    assert cfg.read_text() == "existing = true\n"


def test_init_overwrites_with_force_flag(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("existing = true\n")
    result = runner.invoke(app, ["init", "--config", str(cfg), "--force"])
    assert result.exit_code == 0
    assert "existing" not in cfg.read_text()
