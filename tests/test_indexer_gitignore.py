"""Feature under test: the indexer respects ``.gitignore`` and
``.lookbackignore`` patterns at every level of the walked tree.

Behaviour mirrors ``git`` itself (via the ``pathspec`` library's
``gitignore`` parser): patterns apply to descendants of the
directory they live in, nested ignore files combine, negation
patterns (``!important.log``) un-ignore, and trailing ``/`` matches
directories only.
"""

from __future__ import annotations

from pathlib import Path

from lookback.embed.mock import MockImageEmbedder, MockTextEmbedder
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer
from lookback.schema import IMAGE_EMBED_DIM, TEXT_EMBED_DIM
from lookback.store.lance_store import LanceStore

# Fixture files need real body content under the header — the markdown
# chunker correctly skips header-only files (no chunks to index).
_BODY = "Body content goes here.\n"


def _make_indexer(
    tmp_store_dir: Path, *, respect_gitignore: bool = True
) -> Indexer:
    return Indexer(
        store=LanceStore(tmp_store_dir),
        text_embedder=MockTextEmbedder(dim=TEXT_EMBED_DIM),
        image_embedder=MockImageEmbedder(dim=IMAGE_EMBED_DIM),
        registry=default_registry(),
        respect_gitignore=respect_gitignore,
    )


def _indexed_paths(indexer: Indexer, root: Path) -> set[str]:
    indexer.index_path(root)
    rows = (
        indexer._store.files_table()
        .search()
        .select(["path"])
        .limit(1000)
        .to_list()
    )
    return {r["path"] for r in rows}


def _doc(name: str) -> str:
    return f"# {name}\n{_BODY}"


def test_simple_filename_pattern_is_skipped(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".gitignore").write_text("ignored.md\n")
    (src / "ignored.md").write_text(_doc("ignored"))
    (src / "kept.md").write_text(_doc("kept"))

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert str(src / "kept.md") in paths
    assert str(src / "ignored.md") not in paths


def test_glob_pattern_is_honoured(tmp_path: Path, tmp_store_dir: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".gitignore").write_text("*.log\n")
    (src / "alpha.md").write_text(_doc("alpha"))
    (src / "noisy.log").write_text("log line one\nlog line two\n")
    (src / "other.log").write_text("more lines\nstill more\n")

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert any(p.endswith("alpha.md") for p in paths)
    assert not any(p.endswith(".log") for p in paths)


def test_directory_pattern_skips_whole_subtree(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".gitignore").write_text("private/\n")
    (src / "public.md").write_text(_doc("public"))
    (src / "private").mkdir()
    (src / "private" / "secret.md").write_text(_doc("secret"))
    (src / "private" / "nested" / "deep").mkdir(parents=True)
    (src / "private" / "nested" / "deep" / "deeper.md").write_text(_doc("deeper"))

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert str(src / "public.md") in paths
    # pytest's tmp_path lives under /private/var/... so a substring match on
    # "private" would always fire — pin to the actual subtree instead.
    private_prefix = str(src / "private") + "/"
    assert not any(p.startswith(private_prefix) for p in paths)


def test_negation_pattern_un_ignores(tmp_path: Path, tmp_store_dir: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".gitignore").write_text("*.md\n!important.md\n")
    (src / "boring.md").write_text(_doc("boring"))
    (src / "important.md").write_text(_doc("important"))

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert str(src / "important.md") in paths
    assert str(src / "boring.md") not in paths


def test_nested_gitignore_only_applies_to_its_subtree(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir()
    # Only `a/` has a gitignore; it must not affect siblings.
    (src / "a" / ".gitignore").write_text("draft.md\n")
    (src / "a" / "draft.md").write_text(_doc("a-draft"))
    (src / "a" / "final.md").write_text(_doc("a-final"))
    (src / "b" / "draft.md").write_text(_doc("b-draft"))

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert str(src / "a" / "final.md") in paths
    assert str(src / "b" / "draft.md") in paths
    assert str(src / "a" / "draft.md") not in paths


def test_lookbackignore_is_honoured_alongside_gitignore(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".lookbackignore").write_text("personal.md\n")
    (src / "personal.md").write_text(_doc("personal"))
    (src / "public.md").write_text(_doc("public"))

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert str(src / "public.md") in paths
    assert str(src / "personal.md") not in paths


def test_respect_gitignore_false_indexes_everything(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".gitignore").write_text("normally_ignored.md\n")
    (src / "normally_ignored.md").write_text(_doc("ignored"))
    (src / "always.md").write_text(_doc("always"))

    indexer = _make_indexer(tmp_store_dir, respect_gitignore=False)
    paths = _indexed_paths(indexer, src)
    assert str(src / "normally_ignored.md") in paths
    assert str(src / "always.md") in paths


def test_gitignore_file_itself_is_not_indexed(
    tmp_path: Path, tmp_store_dir: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / ".gitignore").write_text("*.log\n")
    (src / "doc.md").write_text(_doc("doc"))

    paths = _indexed_paths(_make_indexer(tmp_store_dir), src)
    assert all(not p.endswith(".gitignore") for p in paths)
