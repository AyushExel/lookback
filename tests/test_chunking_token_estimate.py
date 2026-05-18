"""Feature under test: the ``estimate_tokens`` helper.

A cheap, deterministic 1-token ≈ 4-character heuristic used everywhere the
indexer needs a rough budget. Real tokenizer counts come from the embedder
when available, but this is what drives chunking decisions.
"""

from __future__ import annotations

import pytest

from lookback.index.chunking import estimate_tokens


def test_empty_string_yields_zero_tokens() -> None:
    assert estimate_tokens("") == 0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a", 1),           # short text still counts as one token
        ("ab", 1),
        ("abcd", 1),
        ("abcde", 1),       # 5 // 4 == 1
        ("abcdefgh", 2),
        ("a" * 40, 10),
        ("a" * 400, 100),
    ],
)
def test_estimate_is_chars_over_four_with_min_one(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected
