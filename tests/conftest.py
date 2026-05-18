"""Cross-cutting pytest fixtures.

Kept deliberately small. Per project rules, feature-specific test files do **not**
share fixtures across unrelated features — only universally useful helpers live here.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_store_dir(tmp_path: Path) -> Iterator[Path]:
    """A throwaway directory for a Lance store, guaranteed empty and auto-removed."""
    d = tmp_path / "store"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)
