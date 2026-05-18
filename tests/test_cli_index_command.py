"""Feature under test: ``lookback index <path>`` walks a directory, runs the
indexer with the configured (mock-by-default) embedders, and reports stats
to stdout. Verifies that the store ends up populated.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lookback.cli import app
from lookback.store.lance_store import LanceStore

runner = CliRunner()


def _write_config(tmp_path: Path, data_dir: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'data_dir = "{data_dir}"\n'
        'text_embedder = "mock"\n'
        'image_embedder = "mock"\n'
    )
    return cfg


def test_index_populates_the_store(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# Title\nbody\n")
    (src / "b.py").write_text("print('hi')\n")
    data_dir = tmp_path / "data"
    cfg = _write_config(tmp_path, data_dir)

    result = runner.invoke(app, ["index", str(src), "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Indexed 2 file(s)" in result.output

    store = LanceStore(data_dir)
    assert store.chunks_text_table().count_rows() >= 2
    assert store.files_table().count_rows() == 2


def test_index_on_missing_path_errors(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, tmp_path / "data")
    result = runner.invoke(
        app, ["index", str(tmp_path / "does-not-exist"), "--config", str(cfg)]
    )
    assert result.exit_code == 2
    assert "not found" in result.output or "not found" in (result.stderr or "")
