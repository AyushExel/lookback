"""Lookback CLI — ``lookback init | index | search | stats | models``.

The CLI is a thin orchestration layer over the store + indexer + embedders.
Every command accepts ``--config`` so tests (and power users) can point at a
non-default config file; without it we fall back to
``~/.lookback/config.toml`` (and, if that's missing, in-memory defaults).
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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
    stats = indexer.index_path(target)
    _print_index_stats(stats)


@app.command()
def search(
    query: str = typer.Argument(..., help="Free-text query."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, help="Max results per modality."),
    modality: str = typer.Option(
        "all",
        "--modality",
        "-m",
        help=(
            "Which embedding space to query: "
            "'text' (text table via the text embedder), "
            "'image' (image table via the joint-space text encoder), "
            "'all' (both — default)."
        ),
    ),
    source_kind: str | None = typer.Option(
        None,
        "--source-kind",
        help="Filter by source kind (markdown, python, screenshot, ...).",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid",
        help=(
            "Fuse full-text search with vector ranking on the text table "
            "(no effect on image queries). Picks up exact-keyword matches "
            "that vector search alone might miss."
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON instead of a table."),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Semantic search across the indexed corpus, text and/or images."""
    if modality not in {"text", "image", "all"}:
        raise typer.BadParameter(
            f"--modality must be one of: text, image, all (got {modality!r})"
        )

    config = LookbackConfig.load(config_path)
    store = LanceStore(config.data_dir)
    where = f"source_kind = '{source_kind}'" if source_kind else None

    text_hits: list[dict] = []
    image_hits: list[dict] = []

    if modality in {"text", "all"}:
        text_embedder = build_text_embedder(config)
        text_query_vec = text_embedder.embed_query(query)
        if hybrid:
            text_hits = store.search_text_hybrid(
                query, text_query_vec, limit=limit, where=where
            )
        else:
            text_hits = store.search_text(text_query_vec, limit=limit, where=where)

    if modality in {"image", "all"}:
        image_text_embedder = build_image_text_embedder(config)
        image_query_vec = image_text_embedder.embed_query(query)
        image_hits = store.search_image(image_query_vec, limit=limit, where=where)

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


def _render_hits_table(hits: list[dict]) -> None:
    table = Table(show_lines=False)
    table.add_column("score", justify="right", style="dim")
    table.add_column("kind", style="cyan")
    table.add_column("meta", style="green", overflow="fold")
    table.add_column("text/file", overflow="fold")

    for h in hits:
        meta = h.get("meta") or ""
        kind = h.get("source_kind", "?")
        text = (h.get("text") or "").replace("\n", " ").strip()[:160]
        # Vector search returns _distance (lower = better); hybrid returns
        # _relevance_score (higher = better). Both render as "score".
        score = h.get("_distance")
        if score is None:
            score = h.get("_relevance_score")
        table.add_row(
            f"{score:.3f}" if isinstance(score, (int, float)) else "",
            kind,
            str(meta)[:80],
            text or "(image)",
        )
    console.print(table)


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
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="MCP transport: 'stdio' (default, used by IDE integrations) or 'http'.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP bind host (HTTP transport only)."),
    port: int = typer.Option(7777, "--port", "-p", help="HTTP port (HTTP transport only)."),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Run the Lookback MCP server.

    Default transport is stdio — that's what Claude Code, Cursor, Continue,
    and most IDE-side MCP clients use. See ``MCP_SETUP.md`` for client
    configuration recipes.
    """
    from lookback.mcp.server import create_server

    config = LookbackConfig.load(config_path)
    server = create_server(config)

    if transport == "stdio":
        # stdio uses stdout for protocol traffic, so log everything else to stderr.
        err_console.print(
            f"[bold]lookback MCP server[/] (stdio) — data_dir={config.data_dir}"
        )
        server.run()
    elif transport == "http":
        err_console.print(
            f"[bold]lookback MCP server[/] (http) — http://{host}:{port}"
        )
        server.run(transport="http", host=host, port=port)
    else:
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
        f"errors={stats.errors}"
    )
    if stats.errors_by_path:
        err_console.print("[yellow]errors:[/]")
        for path, msg in stats.errors_by_path:
            err_console.print(f"  {path}: {msg}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
