"""Lookback CLI — ``lookback init | index | search | stats | models``.

The CLI is a thin orchestration layer over the store + indexer + embedders.
Every command accepts ``--config`` so tests (and power users) can point at a
non-default config file; without it we fall back to
``~/.lookback/config.toml`` (and, if that's missing, in-memory defaults).
"""

from __future__ import annotations

import json as _json
import logging
import os
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lookback.config import LookbackConfig, starter_toml
from lookback.embed.factory import (
    build_image_embedder,
    build_image_text_embedder,
    build_text_embedder,
)
from lookback.embed.models import (
    IMAGE_MODELS,
    TEXT_MODELS,
    image_model,
    text_model,
)
from lookback.extract.registry import default_registry
from lookback.index.indexer import Indexer, IndexStats
from lookback.store.lance_store import LanceStore
from lookback.system import (
    describe_profile,
    detect_profile,
    recommend_image_model,
    recommend_text_model,
)

logger = logging.getLogger("lookback")
app = typer.Typer(
    no_args_is_help=True,
    help="Lookback — local-first multimodal semantic memory for your machine.",
    add_completion=False,
)
models_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and download embedding model weights.",
)
app.add_typer(models_app, name="models")
console = Console()
err_console = Console(stderr=True)


# Typer-friendly enums: typer auto-validates the value at parse time AND
# renders the allowed set in `--help` as e.g. ``[text|image|all]``.
class Modality(StrEnum):
    text = "text"
    image = "image"
    all = "all"


class Transport(StrEnum):
    stdio = "stdio"
    http = "http"


VALID_TEXT_CHOICES = ["mock", *sorted(TEXT_MODELS.keys())]
VALID_IMAGE_CHOICES = ["mock", *sorted(IMAGE_MODELS.keys())]


@app.command()
def init(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Where to write the starter config. Defaults to ~/.lookback/config.toml.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config."),
    text_model_name: str | None = typer.Option(
        None,
        "--text-model",
        help=f"Text embedder. One of: {', '.join(VALID_TEXT_CHOICES)}",
    ),
    image_model_name: str | None = typer.Option(
        None,
        "--image-model",
        help=f"Image embedder. One of: {', '.join(VALID_IMAGE_CHOICES)}",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Prompt for model choices."
    ),
) -> None:
    """Detect the system, recommend models, and write a starter config."""
    profile = detect_profile()
    rec_text = recommend_text_model(profile)
    rec_image = recommend_image_model(profile)

    console.print(f"[bold]Detected:[/] {describe_profile(profile)}")
    console.print(
        f"[bold]Recommended:[/] text=[cyan]{rec_text}[/]  image=[cyan]{rec_image}[/]"
    )

    text_choice = _resolve_model_choice(
        explicit=text_model_name,
        recommended=rec_text,
        valid=VALID_TEXT_CHOICES,
        interactive=interactive,
        kind="text",
    )
    image_choice = _resolve_model_choice(
        explicit=image_model_name,
        recommended=rec_image,
        valid=VALID_IMAGE_CHOICES,
        interactive=interactive,
        kind="image",
    )

    path = Path(config_path).expanduser() if config_path else LookbackConfig.default_path()
    if path.exists() and not force:
        err_console.print(
            f"[yellow]Config already exists at {path}. Pass --force to overwrite.[/]"
        )
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(starter_toml(text_embedder=text_choice, image_embedder=image_choice))
    console.print(f"[green]Wrote starter config to[/] {path}")

    if text_choice != "mock" or image_choice != "mock":
        non_mock: list[str] = []
        if text_choice != "mock":
            non_mock.append(text_choice)
        if image_choice != "mock":
            non_mock.append(image_choice)
        console.print(
            "[bold]Next:[/] download the model weights with "
            f"[cyan]lookback models download {' '.join(non_mock)}[/]"
        )


def _resolve_model_choice(
    *,
    explicit: str | None,
    recommended: str,
    valid: list[str],
    interactive: bool,
    kind: str,
) -> str:
    if explicit is not None:
        if explicit not in valid:
            raise typer.BadParameter(
                f"unknown {kind} model {explicit!r}; expected one of {valid}"
            )
        return explicit
    if interactive:
        choice = typer.prompt(
            f"  Choose {kind} embedder {valid}",
            default=recommended,
        )
        if choice not in valid:
            raise typer.BadParameter(
                f"unknown {kind} model {choice!r}; expected one of {valid}"
            )
        return choice
    return recommended


@app.command()
def index(
    target: Path = typer.Argument(..., help="File or directory to index."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.toml.",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable the live progress display.",
    ),
) -> None:
    """Index a file or directory tree."""
    config = LookbackConfig.load(config_path)
    target = Path(target).expanduser()
    if not target.exists():
        err_console.print(f"[red]path not found:[/] {target}")
        raise typer.Exit(code=2)

    store = LanceStore(config.data_dir)
    text_embedder = build_text_embedder(config)
    image_embedder = build_image_embedder(config)
    indexer = Indexer(
        store=store,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        registry=default_registry(),
        max_file_bytes=config.max_file_bytes,
        skip_hidden=config.skip_hidden,
        follow_symlinks=config.follow_symlinks,
    )

    stats = _run_index_with_progress(indexer, target, disabled=no_progress)
    _print_index_stats(stats)


def _run_index_with_progress(
    indexer: Indexer, target: Path, *, disabled: bool
) -> IndexStats:
    """Run ``indexer.index_path`` with a live spinner + running counts.

    Auto-disables when stdout isn't a TTY (e.g. when output is piped or
    captured by the test harness) so it never pollutes machine-readable
    output.
    """
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    use_progress = not disabled and console.is_terminal
    if not use_progress:
        return indexer.index_path(target)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Indexing[/]"),
        TextColumn(
            "files [green]{task.fields[indexed]}[/] indexed · "
            "[dim]{task.fields[seen]} scanned[/] · "
            "[magenta]{task.fields[chunks]}[/] chunks · "
            "[yellow]{task.fields[flushes]}[/] flushes"
        ),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(
            "idx",
            total=None,
            indexed=0,
            seen=0,
            chunks=0,
            flushes=0,
        )

        def on_file(_path: Path, stats: IndexStats) -> None:
            progress.update(
                task_id,
                indexed=stats.files_indexed,
                seen=stats.files_seen,
                chunks=stats.chunks_written,
                flushes=stats.flushes,
            )

        return indexer.index_path(target, on_file=on_file)


@app.command()
def search(
    query: str = typer.Argument(..., help="Free-text query."),
    limit: int = typer.Option(
        10, "--limit", "-n", min=1, help="Max results per modality."
    ),
    modality: Modality = typer.Option(
        Modality.all,
        "--modality",
        "-m",
        case_sensitive=False,
        help=(
            "Which embedding space to query. "
            "'text' = the text table via your configured text embedder "
            "(default: Nomic v1.5). "
            "'image' = the image table via the joint-space text encoder "
            "(default: MobileCLIP-S2 text tower) — this is how you find "
            "screenshots by describing them. "
            "'all' (default) = run both and show results in two grouped "
            "sections."
        ),
    ),
    source_kind: str | None = typer.Option(
        None,
        "--source-kind",
        help=(
            "Restrict to one extractor's output, e.g. 'markdown', 'python', "
            "'typescript', 'pdf', 'plaintext', 'screenshot'."
        ),
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help=(
            "Fuse full-text search with vector ranking on the text table "
            "(no effect on --modality=image). Surfaces exact-keyword hits "
            "(function names, error codes) that pure vector ranking can "
            "drown out. Ignored when querying only images."
        ),
    ),
    open_top: bool = typer.Option(
        False,
        "--open",
        help=(
            "After printing results, open the top hit with the system "
            "default application (macOS: `open`; Linux: `xdg-open`)."
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit raw JSON instead of a rendered table."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config.toml. Defaults to ~/.lookback/config.toml."
    ),
) -> None:
    """Semantic search across the indexed corpus, text and/or images."""
    config = LookbackConfig.load(config_path)
    store = LanceStore(config.data_dir)
    where = f"source_kind = '{source_kind}'" if source_kind else None

    text_hits: list[dict] = []
    image_hits: list[dict] = []

    if modality in {Modality.text, Modality.all}:
        text_embedder = build_text_embedder(config)
        text_query_vec = text_embedder.embed_query(query)
        if hybrid:
            text_hits = store.search_text_hybrid(
                query, text_query_vec, limit=limit, where=where
            )
        else:
            text_hits = store.search_text(text_query_vec, limit=limit, where=where)

    if modality in {Modality.image, Modality.all}:
        image_text_embedder = build_image_text_embedder(config)
        image_query_vec = image_text_embedder.embed_query(query)
        image_hits = store.search_image(image_query_vec, limit=limit, where=where)

    # Resolve file_id → absolute path so we can render clickable links
    # and so JSON consumers don't need a second roundtrip.
    file_ids = {h["file_id"] for h in (*text_hits, *image_hits) if h.get("file_id")}
    paths_by_id = store.get_files_paths(file_ids) if file_ids else {}
    for h in (*text_hits, *image_hits):
        if h.get("file_id") in paths_by_id:
            h["path"] = paths_by_id[h["file_id"]]

    if json_output:
        console.print_json(
            _json.dumps({"text": text_hits, "image": image_hits}, default=str)
        )
        return

    if not text_hits and not image_hits:
        console.print("[yellow]no results[/]")
        return

    if text_hits:
        console.print("[bold]Text hits[/]")
        _render_hits_table(text_hits)
    if image_hits:
        console.print("[bold]Image hits[/]")
        _render_hits_table(image_hits)
    console.print()
    console.print(
        "[dim]Hint: Cmd-click (Mac) or Ctrl-click (Linux) the path column "
        "to open the file. Use --open to open the top hit automatically.[/]"
    )

    if open_top:
        top_path = (
            (text_hits or image_hits)[0].get("path") if (text_hits or image_hits) else None
        )
        if top_path:
            _open_with_system(Path(top_path))


def _abbreviate_path(path: str) -> str:
    """Replace the user's home dir with ``~`` for a tighter display column."""
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _render_hits_table(hits: list[dict]) -> None:
    table = Table(show_lines=False)
    table.add_column("score", justify="right", style="dim")
    table.add_column("kind", style="cyan", no_wrap=True)
    table.add_column("path", style="green", overflow="fold")
    table.add_column("snippet", overflow="fold")

    for h in hits:
        kind = h.get("source_kind", "?")
        text = (h.get("text") or "").replace("\n", " ").strip()[:200]
        # Vector search returns _distance (lower = better); hybrid returns
        # _relevance_score (higher = better). Both render as "score".
        score = h.get("_distance")
        if score is None:
            score = h.get("_relevance_score")

        full_path = h.get("path")
        if full_path:
            display = _abbreviate_path(full_path)
            # Rich renders this as an OSC-8 hyperlink in supported terminals
            # (Terminal.app, iTerm2, kitty, VS Code, alacritty 0.11+).
            path_cell: object = Text(display, style=f"link file://{full_path}")
        else:
            # Fall back to chunk meta if we couldn't resolve the path.
            meta = h.get("meta") or ""
            path_cell = str(meta)[:80]

        table.add_row(
            f"{score:.3f}" if isinstance(score, (int, float)) else "",
            kind,
            path_cell,
            text or "(image)",
        )
    console.print(table)


def _open_with_system(path: Path) -> None:
    """Open ``path`` with the OS default application."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            err_console.print(f"[yellow]don't know how to open files on {sys.platform}[/]")
    except Exception as exc:
        err_console.print(f"[red]failed to open {path}:[/] {exc}")


@app.command()
def stats(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Print row counts for every table in the store."""
    config = LookbackConfig.load(config_path)
    store = LanceStore(config.data_dir)
    counts = store.stats()
    for name, n in counts.items():
        console.print(f"  {name:14s} [bold]{n:>10d}[/]")


@app.command()
def watch(
    target: Path = typer.Argument(..., help="Directory to watch (recursive)."),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    debounce_ms: int = typer.Option(
        400,
        "--debounce-ms",
        help="Coalesce filesystem events within this many milliseconds before re-indexing.",
    ),
) -> None:
    """Watch a directory and re-index on filesystem changes (Ctrl-C to stop).

    The first pass is *not* run automatically — call ``lookback index`` once
    to seed the store, then ``lookback watch`` to keep it up to date.
    """
    from lookback.index.watcher import Watcher

    config = LookbackConfig.load(config_path)
    target = Path(target).expanduser()
    if not target.exists():
        err_console.print(f"[red]path not found:[/] {target}")
        raise typer.Exit(code=2)

    store = LanceStore(config.data_dir)
    indexer = Indexer(
        store=store,
        text_embedder=build_text_embedder(config),
        image_embedder=build_image_embedder(config),
        registry=default_registry(),
        max_file_bytes=config.max_file_bytes,
        skip_hidden=config.skip_hidden,
        follow_symlinks=config.follow_symlinks,
    )
    watcher = Watcher(indexer, [target], debounce_ms=debounce_ms)
    console.print(f"[bold]Watching[/] {target} (Ctrl-C to stop)")
    try:
        watcher.run()
    except KeyboardInterrupt:
        console.print()
        console.print(
            f"[bold]Stopped[/] — {watcher.stats.batches} batch(es), "
            f"indexed={watcher.stats.files_indexed} "
            f"deleted={watcher.stats.files_deleted} "
            f"errors={watcher.stats.errors}"
        )


@app.command()
def serve(
    transport: Transport = typer.Option(
        Transport.stdio,
        "--transport",
        "-t",
        case_sensitive=False,
        help=(
            "MCP transport. 'stdio' (default) is what Claude Code, Cursor, "
            "Continue, ChatGPT Desktop, Windsurf, and Zed use — the client "
            "spawns the server and talks over stdin/stdout. 'http' binds an "
            "HTTP server for cross-machine setups."
        ),
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="HTTP bind host. Only used when --transport=http."
    ),
    port: int = typer.Option(
        7777, "--port", "-p", help="HTTP port. Only used when --transport=http."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config.toml. Defaults to ~/.lookback/config.toml."
    ),
) -> None:
    """Run the Lookback MCP server.

    Default transport is stdio — that's what Claude Code, Cursor, Continue,
    and most IDE-side MCP clients use. See ``MCP_SETUP.md`` for client
    configuration recipes.
    """
    from lookback.mcp.server import create_server

    config = LookbackConfig.load(config_path)
    server = create_server(config)

    if transport is Transport.stdio:
        # stdio uses stdout for protocol traffic, so log everything else to stderr.
        err_console.print(
            f"[bold]lookback MCP server[/] (stdio) — data_dir={config.data_dir}"
        )
        server.run()
    elif transport is Transport.http:
        err_console.print(
            f"[bold]lookback MCP server[/] (http) — http://{host}:{port}"
        )
        server.run(transport="http", host=host, port=port)
    else:  # pragma: no cover — enum exhausted above
        raise typer.BadParameter(
            f"--transport must be 'stdio' or 'http' (got {transport!r})"
        )


@models_app.command("list")
def models_list() -> None:
    """Print every registered text and image embedding model."""
    console.print("[bold]Text models[/]")
    for name, spec in TEXT_MODELS.items():
        console.print(
            f"  [cyan]{name}[/]  dim={spec.dim}  disk≈{spec.approx_disk_mb} MB  "
            f"min_ram={spec.min_ram_gb} GB"
        )
        console.print(f"    {spec.description}")
        console.print(f"    repo: {spec.hf_repo}")
    console.print()
    console.print("[bold]Image models[/]")
    for name, spec in IMAGE_MODELS.items():
        console.print(
            f"  [cyan]{name}[/]  dim={spec.dim}  image_size={spec.image_size}  "
            f"disk≈{spec.approx_disk_mb} MB"
        )
        console.print(f"    {spec.description}")
        console.print(f"    repo: {spec.hf_repo}")


@models_app.command("download")
def models_download(
    names: list[str] = typer.Argument(
        ...,
        help="One or more model names (e.g. nomic-v1.5 mobileclip-s2).",
    ),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Download model weights from Hugging Face into ``models_dir``."""
    config = LookbackConfig.load(config_path)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    from lookback.embed.download import (
        download_image_model,
        download_text_model,
        image_model_target_dir,
        text_model_target_dir,
    )

    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            if name in TEXT_MODELS:
                spec = text_model(name)
                target = text_model_target_dir(config.models_dir, name)
                console.print(
                    f"[bold]Downloading[/] [cyan]{spec.display_name}[/] → {target}"
                )
                paths = download_text_model(spec, target)
                console.print(f"  model     → {paths.model}")
                console.print(f"  tokenizer → {paths.tokenizer}")
            elif name in IMAGE_MODELS:
                spec = image_model(name)
                target = image_model_target_dir(config.models_dir, name)
                console.print(
                    f"[bold]Downloading[/] [cyan]{spec.display_name}[/] → {target}"
                )
                paths = download_image_model(spec, target)
                console.print(f"  vision    → {paths.vision}")
                if paths.text is not None:
                    console.print(f"  text      → {paths.text}")
                if paths.tokenizer is not None:
                    console.print(f"  tokenizer → {paths.tokenizer}")
            else:
                failures.append((name, "unknown model — run `lookback models list`"))
                err_console.print(f"[red]unknown model[/] {name!r}")
        except Exception as exc:
            failures.append((name, str(exc)))
            err_console.print(f"[red]download failed[/] {name}: {exc}")

    if failures:
        err_console.print()
        err_console.print(f"[yellow]Completed with {len(failures)} failure(s):[/]")
        for name, msg in failures:
            err_console.print(f"  [red]{name}[/]: {msg}")
        raise typer.Exit(code=1)




def _print_index_stats(stats: IndexStats) -> None:
    console.print(
        f"[green]Indexed[/] {stats.files_indexed} file(s), "
        f"[green]wrote[/] {stats.chunks_written} chunk(s); "
        f"unchanged={stats.files_unchanged} skipped={stats.files_skipped} "
        f"errors={stats.errors} · "
        f"[dim]flushes={stats.flushes} optimizations={stats.optimizations}[/]"
    )
    if stats.errors_by_path:
        err_console.print("[yellow]errors:[/]")
        for path, msg in stats.errors_by_path:
            err_console.print(f"  {path}: {msg}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
