"""Code extractor — fixed line-window chunking, CODE modality.

``source_kind`` is the inferred language (e.g. ``"python"``, ``"typescript"``)
so callers can filter searches to a single language with a scalar predicate.
"""

from __future__ import annotations

from pathlib import Path

from lookback.extract.base import ExtractedChunk, Extractor
from lookback.index.chunking import chunk_code
from lookback.schema import Modality

LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".lua": "lua",
    ".r": "r",
    ".jl": "julia",
    ".php": "php",
    ".scala": "scala",
    ".clj": "clojure",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".pl": "perl",
    ".dart": "dart",
    ".sql": "sql",
}


class CodeExtractor(Extractor):
    extensions = frozenset(LANGUAGE_BY_EXT.keys())

    def extract(self, path: Path) -> list[ExtractedChunk]:
        ext = path.suffix.lower()
        language = LANGUAGE_BY_EXT.get(ext, "unknown")
        content = path.read_text(encoding="utf-8", errors="replace")
        return [
            ExtractedChunk(
                text=c.text,
                modality=Modality.CODE,
                source_kind=language,
                meta=c.meta,
            )
            for c in chunk_code(content, language=language)
        ]
