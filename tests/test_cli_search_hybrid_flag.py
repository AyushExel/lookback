"""Feature under test: ``lookback search --hybrid`` routes the text-table
query through ``search_text_hybrid`` so callers get FTS + vector RRF
fusion. Image queries are unaffected.

We assert the *behavioural* difference (hybrid returns hits with the
``_relevance_score`` field; vector-only returns ``_distance``) rather than
poking at internals.
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


def _seed(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# IVF_PQ tuning\nDetails about IVF_PQ partitions.\n")
    (src / "b.md").write_text("# Pasta\nCarbonara recipe.\n")
    return src


def test_hybrid_flag_returns_relevance_score(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    src = _seed(tmp_path)
    assert runner.invoke(app, ["index", str(src), "--config", str(cfg)]).exit_code == 0

    result = runner.invoke(
        app,
        [
            "search",
            "IVF_PQ tuning",
            "--config",
            str(cfg),
            "--modality",
            "text",
            "--hybrid",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output.strip())
    assert data["text"], "expected at least one hybrid hit"
    for hit in data["text"]:
        assert "_relevance_score" in hit, (
            f"hybrid hit should carry _relevance_score; got keys: {list(hit)}"
        )
        # Vector's _distance field must not appear in hybrid results.
        assert "_distance" not in hit


def test_no_hybrid_flag_returns_distance(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    src = _seed(tmp_path)
    assert runner.invoke(app, ["index", str(src), "--config", str(cfg)]).exit_code == 0

    result = runner.invoke(
        app,
        ["search", "anything", "--config", str(cfg), "--modality", "text", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output.strip())
    assert data["text"]
    for hit in data["text"]:
        assert "_distance" in hit
        assert "_relevance_score" not in hit
