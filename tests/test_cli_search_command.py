"""Feature under test: ``lookback search <query>`` embeds the query with the
configured embedder, runs a vector search against ``chunks_text``, and
prints results (table or JSON).

We seed the store with a small mock-embedded corpus, then verify the CLI
returns at least one hit and respects ``--limit``.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app

runner = CliRunner()


def _write_config(tmp_path: Path, data_dir: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'data_dir = "{data_dir}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    return cfg


def _seed_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.md").write_text("# Attention\nNotes on transformer attention.\n")
    (src / "lance.md").write_text("# LanceDB\nNotes on IVF_PQ tuning.\n")
    (src / "rust.md").write_text("# Rust\nOwnership semantics.\n")
    return src


def test_search_returns_results_in_table_form(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, tmp_path / "data")
    src = _seed_corpus(tmp_path)
    assert runner.invoke(app, ["index", str(src), "--config", str(cfg)]).exit_code == 0

    result = runner.invoke(
        app,
        ["search", "transformer attention notes", "--config", str(cfg), "--limit", "3"],
    )
    assert result.exit_code == 0, result.output
    # Either a table header or the JSON-less fallback "no results" must NOT appear.
    assert "no results" not in result.output


def test_search_json_output_is_parseable_json(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, tmp_path / "data")
    src = _seed_corpus(tmp_path)
    assert runner.invoke(app, ["index", str(src), "--config", str(cfg)]).exit_code == 0

    result = runner.invoke(
        app,
        ["search", "anything", "--config", str(cfg), "--json", "--limit", "5"],
    )
    assert result.exit_code == 0, result.output
    # JSON output is now keyed by modality so the consumer can tell text and
    # image hits apart in cross-modal results.
    data = _json.loads(result.output.strip())
    assert isinstance(data, dict)
    assert set(data.keys()) == {"text", "image"}
    assert isinstance(data["text"], list)
    assert isinstance(data["image"], list)
    assert len(data["text"]) <= 5
    assert len(data["image"]) <= 5
