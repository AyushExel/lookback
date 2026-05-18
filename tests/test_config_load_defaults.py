"""Feature under test: ``LookbackConfig.load`` returns sensible defaults when
no config file is present, and exposes a stable ``default_path``.
"""

from __future__ import annotations

from pathlib import Path

from lookback.config import DEFAULT_HOME, LookbackConfig


def test_default_path_is_under_home() -> None:
    assert LookbackConfig.default_path() == DEFAULT_HOME / "config.toml"


def test_load_with_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = LookbackConfig.load(tmp_path / "nonexistent.toml")
    assert cfg.data_dir == DEFAULT_HOME / "data"
    assert cfg.roots == []
    assert cfg.skip_hidden is True
    assert cfg.follow_symlinks is False
    assert cfg.max_file_bytes == 50 * 1024 * 1024
    assert cfg.text_embedder == "mock"
    assert cfg.image_embedder == "mock"
