"""Feature under test: ``lookback init`` honours ``--text-model`` and
``--image-model`` overrides, falling back to system recommendations when
the flags are absent, and rejects unknown model names with a clear error.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app
from lookback.config import LookbackConfig

runner = CliRunner()


def test_explicit_text_and_image_model_flags_land_in_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        [
            "init",
            "--config",
            str(cfg),
            "--text-model",
            "nomic-v1.5",
            "--image-model",
            "mobileclip-s2",
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = LookbackConfig.load(cfg)
    assert loaded.text_embedder == "nomic-v1.5"
    assert loaded.image_embedder == "mobileclip-s2"


def test_no_flags_uses_recommended_models(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    result = runner.invoke(app, ["init", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    loaded = LookbackConfig.load(cfg)
    # Recommendations must point at registered models or the safe 'mock' fallback.
    assert loaded.text_embedder in {"mock", "nomic-v1.5", "nomic-v2-moe"}
    assert loaded.image_embedder in {"mock", "mobileclip-s2"}


def test_unknown_text_model_flag_errors(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        ["init", "--config", str(cfg), "--text-model", "not-a-real-model"],
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "not-a-real-model" in combined


def test_next_step_message_appears_for_real_models(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        [
            "init",
            "--config",
            str(cfg),
            "--text-model",
            "nomic-v1.5",
            "--image-model",
            "mobileclip-s2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "lookback models download" in result.output


def test_next_step_message_absent_for_mock_only(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        [
            "init",
            "--config",
            str(cfg),
            "--text-model",
            "mock",
            "--image-model",
            "mock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "models download" not in result.output
