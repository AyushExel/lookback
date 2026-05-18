"""Feature under test: ``lookback search --modality {text|image|all}`` routes
the query through the right embedding space.

With mock embedders the *semantic* ranking isn't meaningful, but the
dispatch path is — these tests verify that
- ``--modality text`` returns only text-table hits and zero image-table hits,
- ``--modality image`` returns only image-table hits,
- the default ``all`` returns from both,
- an invalid ``--modality`` value errors out cleanly.

A separate ``@needs_models`` test exercises real-weight cross-modal quality.
"""

from __future__ import annotations

import json as _json
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


def _seed_mixed_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.md").write_text("# Notes\nA paragraph of words.\n")
    (src / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return src


def _index(cfg: Path, src: Path) -> None:
    assert runner.invoke(app, ["index", str(src), "--config", str(cfg)]).exit_code == 0


def test_modality_text_returns_only_text_hits(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _index(cfg, _seed_mixed_corpus(tmp_path))
    result = runner.invoke(
        app,
        ["search", "anything", "--config", str(cfg), "--modality", "text", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output.strip())
    assert data["text"], "expected at least one text hit"
    assert data["image"] == [], "text modality must not query the image table"


def test_modality_image_returns_only_image_hits(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _index(cfg, _seed_mixed_corpus(tmp_path))
    result = runner.invoke(
        app,
        ["search", "anything", "--config", str(cfg), "--modality", "image", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output.strip())
    assert data["image"], "expected at least one image hit"
    assert data["text"] == [], "image modality must not query the text table"


def test_default_modality_queries_both_tables(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _index(cfg, _seed_mixed_corpus(tmp_path))
    result = runner.invoke(
        app,
        ["search", "anything", "--config", str(cfg), "--json"],
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output.strip())
    assert data["text"] and data["image"], (
        "default --modality should return hits from both tables"
    )


def test_invalid_modality_value_errors(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _index(cfg, _seed_mixed_corpus(tmp_path))
    result = runner.invoke(
        app,
        ["search", "x", "--config", str(cfg), "--modality", "nope"],
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "modality" in combined.lower()
