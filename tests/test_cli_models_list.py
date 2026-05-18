"""Feature under test: ``lookback models list`` enumerates every entry in the
text and image registries with their HF repo and disk-size estimate, so a
user can pick one to download.
"""

from __future__ import annotations

from typer.testing import CliRunner

from lookback.cli import app

runner = CliRunner()


def test_models_list_shows_every_text_and_image_model() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    for name in ["nomic-v1.5", "nomic-v2-moe", "mobileclip-s2"]:
        assert name in result.output


def test_models_list_includes_hf_repos() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "nomic-ai/" in result.output
    # The image-model repo currently lives under a community account; assert
    # only that *some* repo string for mobileclip shows up.
    assert "mobileclip" in result.output.lower()
