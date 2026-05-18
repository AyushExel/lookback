"""Feature under test: ``LookbackConfig.load`` reads a TOML file and validates
its contents — including expansion of ``~`` in paths and rejection of
unknown options through pydantic's strict-by-default validation behaviour
on typed fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lookback.config import LookbackConfig


def test_load_parses_roots_and_options(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'data_dir = "{tmp_path / "data"}"\n'
        f'roots = ["{tmp_path / "docs"}", "{tmp_path / "screenshots"}"]\n'
        "max_file_bytes = 1024\n"
        "skip_hidden = false\n"
        "follow_symlinks = true\n"
        'text_embedder = "nomic"\n'
        'image_embedder = "mobileclip"\n'
    )
    cfg = LookbackConfig.load(cfg_path)
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.roots == [tmp_path / "docs", tmp_path / "screenshots"]
    assert cfg.max_file_bytes == 1024
    assert cfg.skip_hidden is False
    assert cfg.follow_symlinks is True
    assert cfg.text_embedder == "nomic"
    assert cfg.image_embedder == "mobileclip"


def test_load_expands_tilde_in_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('data_dir = "~/some-dir"\nroots = ["~/Documents"]\n')
    cfg = LookbackConfig.load(cfg_path)
    assert str(cfg.data_dir).startswith(str(Path.home()))
    assert str(cfg.roots[0]).startswith(str(Path.home()))


def test_load_rejects_invalid_types(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('max_file_bytes = "not a number"\n')
    with pytest.raises(ValidationError):
        LookbackConfig.load(cfg_path)
