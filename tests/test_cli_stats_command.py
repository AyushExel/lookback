"""Feature under test: ``lookback stats`` prints a row count per table.

After indexing a fixture corpus, the printed output must include every
table name and a non-zero count for at least the text-chunks table.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app
from lookback.store.lance_store import CHUNKS_IMAGE, CHUNKS_TEXT, FILES

runner = CliRunner()


def test_stats_lists_every_table(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.md").write_text("# Title\nbody\n")
    runner.invoke(app, ["index", str(src), "--config", str(cfg)])

    result = runner.invoke(app, ["stats", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    for name in (CHUNKS_TEXT, CHUNKS_IMAGE, FILES):
        assert name in result.output
