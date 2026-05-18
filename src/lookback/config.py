"""Config — pydantic model + TOML loader for ``~/.lookback/config.toml``.

Everything has a sensible default so a brand-new install is usable without a
config file. ``lookback init`` writes a starter file the user can edit.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DEFAULT_HOME = Path.home() / ".lookback"


class LookbackConfig(BaseModel):
    data_dir: Path = Field(default_factory=lambda: DEFAULT_HOME / "data")
    models_dir: Path = Field(default_factory=lambda: DEFAULT_HOME / "models")
    roots: list[Path] = Field(default_factory=list)

    max_file_bytes: int = 50 * 1024 * 1024
    skip_hidden: bool = True
    follow_symlinks: bool = False

    text_embedder: str = "mock"
    image_embedder: str = "mock"

    @field_validator("data_dir", "models_dir", mode="before")
    @classmethod
    def _expand_path_field(cls, v: Path | str) -> Path:
        return Path(v).expanduser()

    @field_validator("roots", mode="before")
    @classmethod
    def _expand_roots(cls, v: list[Path | str]) -> list[Path]:
        return [Path(p).expanduser() for p in v]

    @classmethod
    def default_path(cls) -> Path:
        return DEFAULT_HOME / "config.toml"

    @classmethod
    def load(cls, path: Path | None = None) -> LookbackConfig:
        path = Path(path) if path else cls.default_path()
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)


def starter_toml(
    roots: list[Path] | None = None,
    *,
    text_embedder: str = "mock",
    image_embedder: str = "mock",
    data_dir: Path | None = None,
    models_dir: Path | None = None,
) -> str:
    """Generate the contents of a starter ``config.toml``.

    The stdlib reads TOML but doesn't write it, so we hand-format. The
    embedder names default to ``"mock"`` so a fresh install works without
    downloading model weights — users (and ``lookback init``) overwrite
    them once they pick real models.
    """
    roots = roots or [Path.home() / "Documents", Path.home() / "Pictures" / "Screenshots"]
    data_dir = data_dir or DEFAULT_HOME / "data"
    models_dir = models_dir or DEFAULT_HOME / "models"
    roots_block = "\n".join(f'  "{p}",' for p in roots)
    return (
        "# ~/.lookback/config.toml — Lookback configuration\n"
        "#\n"
        "# Edit this file to add directories you want indexed. After saving,\n"
        "# run `lookback index` to (re-)index them.\n"
        "\n"
        f'data_dir = "{data_dir}"\n'
        f'models_dir = "{models_dir}"\n'
        "\n"
        "# Directories to index. Add or remove freely.\n"
        "roots = [\n"
        f"{roots_block}\n"
        "]\n"
        "\n"
        "# Skip files larger than this many bytes (default 50 MiB).\n"
        "max_file_bytes = 52428800\n"
        "\n"
        "# Skip hidden files and dotted directories.\n"
        "skip_hidden = true\n"
        "follow_symlinks = false\n"
        "\n"
        "# Embedder selection. 'mock' is for testing; for production use\n"
        "# pick a real model and run `lookback models download <name>`.\n"
        f'text_embedder = "{text_embedder}"\n'
        f'image_embedder = "{image_embedder}"\n'
    )
