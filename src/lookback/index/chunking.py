"""Text chunking strategies for the indexer.

Three strategies live here: ``chunk_markdown`` (header-aware), ``chunk_plaintext``
(paragraph-aware with token-budget windows), and ``chunk_code`` (fixed line
windows with line-count overlap). They share the simple ``Chunk`` payload and a
shared 1-token ≈ 4-characters approximation that's good enough for budgeting
chunks at index time — the exact tokenizer for the embedder is consulted later
if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """1 token ≈ 4 characters. Cheap, deterministic, never zero for non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_markdown(
    content: str,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split a markdown document by ATX headers, then by token budget within sections.

    Each output chunk carries the deepest enclosing header (or ``None`` for
    pre-header preamble) in ``meta["section"]``, along with a 1-based
    ``meta["section_idx"]`` so adjacent same-section chunks can be told apart.
    """
    sections = _split_markdown_sections(content)
    chunks: list[Chunk] = []
    for section_idx, (header, body) in enumerate(sections, start=1):
        if not body.strip():
            continue
        section_meta = {"section": header, "section_idx": section_idx}
        for piece in _budget_split(
            body, target_tokens=target_tokens, overlap_tokens=overlap_tokens
        ):
            chunks.append(Chunk(text=piece, meta=dict(section_meta)))
    return chunks


def _split_markdown_sections(content: str) -> list[tuple[str | None, str]]:
    """Return ``[(header_or_None, body), ...]`` in document order."""
    matches = list(_HEADER_RE.finditer(content))
    if not matches:
        return [(None, content)]
    sections: list[tuple[str | None, str]] = []
    first = matches[0]
    if first.start() > 0:
        sections.append((None, content[: first.start()]))
    for i, m in enumerate(matches):
        header_text = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((header_text, content[body_start:body_end]))
    return sections


def chunk_plaintext(
    content: str,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Paragraph-aware windowing. Splits on blank lines first, then by token budget."""
    if not content.strip():
        return []
    chunks: list[Chunk] = []
    for piece in _budget_split(
        content, target_tokens=target_tokens, overlap_tokens=overlap_tokens
    ):
        chunks.append(Chunk(text=piece, meta={}))
    return chunks


def chunk_code(
    content: str,
    *,
    target_lines: int = 40,
    overlap_lines: int = 8,
    language: str | None = None,
) -> list[Chunk]:
    """Fixed line-window chunking with line-count overlap.

    Stores ``meta["line_start"]`` and ``meta["line_end"]`` (1-based, inclusive)
    plus the language tag the extractor inferred.
    """
    if target_lines <= 0:
        raise ValueError("target_lines must be positive")
    if overlap_lines < 0 or overlap_lines >= target_lines:
        raise ValueError("overlap_lines must satisfy 0 <= overlap < target")

    lines = content.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    step = target_lines - overlap_lines
    start = 0
    while start < len(lines):
        end = min(start + target_lines, len(lines))
        window = lines[start:end]
        text = "\n".join(window)
        if text.strip():
            chunks.append(
                Chunk(
                    text=text,
                    meta={
                        "line_start": start + 1,
                        "line_end": end,
                        "language": language,
                    },
                )
            )
        if end >= len(lines):
            break
        start += step
    return chunks


def _budget_split(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks of approximately ``target_tokens`` tokens.

    Preserves paragraph boundaries (blank-line separators) where possible. If
    a paragraph alone exceeds the budget, falls back to character windows of
    ``target_tokens * 4`` chars with ``overlap_tokens * 4`` char overlap.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < target")

    target_chars = target_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    if estimate_tokens(text) <= target_tokens:
        return [text.strip()] if text.strip() else []

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    for para in paragraphs:
        para_chars = len(para)
        if para_chars > target_chars:
            if buf:
                out.append("\n\n".join(buf).strip())
                buf = []
                buf_chars = 0
            out.extend(_char_window(para, target_chars, overlap_chars))
            continue
        if buf_chars + para_chars + 2 > target_chars and buf:
            out.append("\n\n".join(buf).strip())
            # Carry the last paragraph as overlap if it fits.
            if buf[-1] and len(buf[-1]) <= overlap_chars:
                buf = [buf[-1]]
                buf_chars = len(buf[-1])
            else:
                buf = []
                buf_chars = 0
        buf.append(para)
        buf_chars += para_chars + 2
    if buf:
        out.append("\n\n".join(buf).strip())
    return [c for c in out if c]


def _char_window(text: str, window_chars: int, overlap_chars: int) -> list[str]:
    step = window_chars - overlap_chars
    out = []
    start = 0
    while start < len(text):
        end = min(start + window_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start += step
    return out
