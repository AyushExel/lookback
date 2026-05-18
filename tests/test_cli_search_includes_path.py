"""Feature under test: ``lookback search`` resolves the file's absolute
path for each hit and includes it in JSON output (and, via the table
renderer, as a clickable ``file://`` link in supported terminals).

Without this, a user staring at ``{"filename": "Screenshot 2026-02-13...png"}``
had no way to know which folder the file was in or to open it. We
now do a single batched lookup against the ``files`` table per query
and stitch the path onto every hit dict.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app

runner = CliRunner()


def _seed(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.md").write_text("# Title\nBody content.\n")
    runner.invoke(app, ["index", str(src), "--config", str(cfg), "--no-progress"])
    return cfg


def test_search_json_includes_path_field(tmp_path: Path) -> None:
    cfg = _seed(tmp_path)
    result = runner.invoke(
        app, ["search", "anything", "--config", str(cfg), "--json", "--modality", "text"]
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output.strip())
    assert data["text"], "expected at least one text hit"
    hit = data["text"][0]
    assert "path" in hit, f"missing 'path' field in hit; got keys {list(hit)}"
    assert hit["path"].endswith("doc.md")
