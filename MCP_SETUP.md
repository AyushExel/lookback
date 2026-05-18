# Connecting Lookback to your AI tools (MCP setup)

Lookback ships an [MCP](https://modelcontextprotocol.io) server so any
MCP-capable AI tool — Claude Code, Cursor, Continue, ChatGPT Desktop,
Windsurf, Zed — can call into your local index as a tool. Once wired up,
your assistant can answer questions like *"find that diagram I screenshotted
last week"* by calling `lookback_search` on its own, without you opening a
new terminal.

## TL;DR

1. Install lookback: `pip install lookback-ai` (or for development: `uv sync && uv tool install --editable .`).
2. Make sure the index is populated: `lookback init && lookback models download nomic-v1.5 mobileclip-s2 && lookback index ~/Documents ~/Pictures/Screenshots`.
3. Paste the JSON snippet for your client below into its MCP config.
4. Restart the client. The `lookback_search` tool will be available.

## What gets exposed

The server registers two tools:

| Tool | Description |
|---|---|
| `lookback_search(query, modality, limit, source_kind)` | Semantic search across the user's locally indexed files. Returns `{"text": [...], "image": [...]}`. `modality` is `"text"`, `"image"`, or `"all"` (default). |
| `lookback_stats()` | Row counts per table — useful for "is my index populated?" diagnostics. |

Both tools run fully on your machine. **No data leaves your laptop.** The MCP
client (Claude Code, Cursor, etc.) sees the tool *results* and can show them
to you or use them as context — but the underlying files and embeddings stay
local.

## Transports

| Transport | When to use |
|---|---|
| **stdio** (default) | Every IDE-side MCP client (Claude Code, Cursor, Continue, Windsurf, Zed, ChatGPT Desktop). The client spawns `lookback serve` and talks to it over stdin/stdout. **This is what you almost certainly want.** |
| **HTTP** | Hosted scenarios — you're running Lookback on one machine and an AI assistant elsewhere. Bind with `lookback serve --transport http --port 7777`. |

## Claude Code

Claude Code reads MCP servers from `.mcp.json` in the workspace root (for
project-scoped servers) or `~/.claude.json` (for user-scoped servers).

Add this to whichever you prefer:

```jsonc
{
  "mcpServers": {
    "lookback": {
      "command": "uv",
      "args": ["run", "lookback", "serve"],
      "cwd": "/absolute/path/to/your/lookback/checkout"
    }
  }
}
```

If you installed lookback globally via pipx or `pip install --user`, drop
the `uv run` wrapper:

```jsonc
{
  "mcpServers": {
    "lookback": {
      "command": "lookback",
      "args": ["serve"]
    }
  }
}
```

After saving, restart Claude Code. The `lookback_search` tool will show up in
the tools panel. You can confirm with `/mcp` inside Claude Code — `lookback`
should appear as a connected server.

## Cursor

Cursor uses `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` in the
workspace root (project-scoped). Same schema as Claude Code:

```jsonc
{
  "mcpServers": {
    "lookback": {
      "command": "lookback",
      "args": ["serve"]
    }
  }
}
```

Open the Cursor command palette → `MCP: List Servers` to confirm.

## Continue (VS Code / JetBrains)

In your `~/.continue/config.json` (or `config.yaml`):

```jsonc
{
  "mcpServers": [
    {
      "name": "lookback",
      "command": "lookback",
      "args": ["serve"]
    }
  ]
}
```

## ChatGPT Desktop (and other Anthropic / OpenAI clients)

The desktop apps that support MCP read from `~/Library/Application Support/<App>/mcp.json` on macOS
(or the platform equivalent). The schema mirrors the others:

```jsonc
{
  "mcpServers": {
    "lookback": {
      "command": "lookback",
      "args": ["serve"]
    }
  }
}
```

## Windsurf

Windsurf's MCP config lives at `~/.codeium/windsurf/mcp_config.json`:

```jsonc
{
  "mcpServers": {
    "lookback": {
      "command": "lookback",
      "args": ["serve"]
    }
  }
}
```

## Zed

Zed reads MCP configuration from its settings. Add to `~/.config/zed/settings.json`:

```jsonc
{
  "context_servers": {
    "lookback": {
      "command": {
        "path": "lookback",
        "args": ["serve"]
      }
    }
  }
}
```

## HTTP transport (for remote / multi-machine setups)

If your AI client doesn't support stdio (rare in 2026) or you want to run
Lookback on a different machine than your assistant:

```bash
lookback serve --transport http --host 0.0.0.0 --port 7777
```

Configure the client to point at `http://your-host:7777`. Most MCP clients
default-trust localhost; for cross-machine usage put it behind your usual
TLS / auth layer — Lookback itself does not implement either.

## Sample queries

Once connected, you can ask your assistant things like:

- *"Use lookback to find anything I've written about IVF_PQ tuning."* → calls `lookback_search("IVF_PQ tuning")`.
- *"Search my screenshots for a chart with red and blue lines."* → calls `lookback_search("chart with red and blue lines", modality="image")`.
- *"How many things are in my Lookback index?"* → calls `lookback_stats()`.

Most assistants will pick the right tool and parameters automatically once
the server is connected. If yours doesn't, mention "use lookback" in the
prompt to nudge it.

## Troubleshooting

**The tool doesn't appear in my client.**
- Restart the client after editing the MCP config.
- Run `lookback serve` directly in a terminal — if it errors there, that's the same error the client sees.
- Check the client's MCP log. Claude Code: `/mcp` shows status per server.

**`lookback_search` returns empty results.**
- Run `lookback stats --config <your-config>` to confirm the index is populated. If counts are 0, run `lookback index <paths>` first.
- The config path the MCP server uses is `~/.lookback/config.toml` unless you pass `--config` explicitly. Make sure the client invocation isn't pointing at an empty config.

**Wrong embedder, dimension errors, etc.**
- The MCP server reads from the same config as the CLI. If you ran `lookback init --text-model nomic-v1.5` and then accidentally deleted the model files, the server will fail loudly. Re-run `lookback models download nomic-v1.5 mobileclip-s2`.

**Performance.**
- First search after server start is slow — Nomic + MobileCLIP ONNX sessions load lazily on first call. Steady-state is fast (≤200ms per query on a typical M-series Mac for an index under ~1M chunks).
- For very large indexes, `lookback compact` periodically (currently called via `LanceStore.optimize()` programmatically; CLI subcommand pending).

## Custom config paths

If you keep multiple Lookback indexes (e.g. a personal one and a work one),
pass `--config /path/to/config.toml` to `lookback serve`:

```jsonc
{
  "mcpServers": {
    "lookback-work": {
      "command": "lookback",
      "args": ["serve", "--config", "/Users/me/work/lookback.toml"]
    },
    "lookback-personal": {
      "command": "lookback",
      "args": ["serve", "--config", "/Users/me/personal/lookback.toml"]
    }
  }
}
```

Each shows up as a distinct server with its own tools (`lookback-work__lookback_search`, etc., depending on the client's namespacing rules).
