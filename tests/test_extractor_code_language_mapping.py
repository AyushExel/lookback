"""Feature under test: ``CodeExtractor`` maps a file extension to the right
``source_kind`` (language tag) on every emitted chunk.

The language tag is later used as a bitmap-indexed scalar filter, so it has
to be stable and lowercase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lookback.extract.code import CodeExtractor
from lookback.schema import Modality


@pytest.mark.parametrize(
    "ext,language",
    [
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".go", "go"),
        (".rs", "rust"),
        (".java", "java"),
        (".sh", "shell"),
        (".sql", "sql"),
    ],
)
def test_extension_maps_to_language(ext: str, language: str, tmp_path: Path) -> None:
    f = tmp_path / f"sample{ext}"
    f.write_text("print('hello')\nprint('world')\n")
    chunks = CodeExtractor().extract(f)
    assert chunks, f"expected chunks for {ext}"
    for c in chunks:
        assert c.source_kind == language
        assert c.modality is Modality.CODE
        assert c.meta.get("language") == language
