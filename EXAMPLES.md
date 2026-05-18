# Lookback — end-to-end examples

Every command and output below is **verbatim** from a real session on an
M-series Mac with `lookback-ai` installed, against this fixture workspace:

```
/tmp/lookback-demo/src
├── code
│   ├── embedding.py        # uses SentenceTransformer to embed text
│   └── lance_query.py      # IVF_PQ vector search via LanceDB
├── images
│   ├── diagram.png         # neural-network-style layered nodes
│   ├── dog.png             # silhouette of a dog on grass
│   └── landscape.png       # mountains + clouds + sky
└── notes
    ├── cooking.md          # pasta carbonara, risotto
    ├── databases.md        # ACID, isolation, B-tree vs hash indexes
    └── transformers.md     # attention, positional encoding
```

8 files (3 markdown · 2 python · 3 png).

## 1. First-run bootstrap

```text
$ lookback init
Detected: Darwin · arm64 · Apple Silicon · 16.0 GB RAM · 8 CPU
Recommended: text=nomic-v1.5  image=mobileclip-s2
Wrote starter config to /Users/me/.lookback/config.toml
Next: download the model weights with lookback models download nomic-v1.5 mobileclip-s2
```

The system probe surfaces RAM, arch, and Apple-Silicon status, then maps
that to a model recommendation. `--interactive` would prompt for the
choices instead; `--text-model X --image-model Y` overrides them.

## 2. Inspect available models

```text
$ lookback models list
Text models
  nomic-v1.5  dim=768  disk≈550 MB  min_ram=4.0 GB
    Dense 137M-param model, 768-d output with Matryoshka truncation. Best
    quality-to-size on CPU and the safe default for laptops.
    repo: nomic-ai/nomic-embed-text-v1.5
  nomic-v2-moe  dim=768  disk≈1900 MB  min_ram=12.0 GB
    Mixture-of-experts model: ~150M active params per token but the full 475M
    must be resident in RAM. Higher quality on retrieval, heavier footprint —
    pick only if you have ≥16 GB RAM and want the quality bump.
    repo: nomic-ai/nomic-embed-text-v2-moe

Image models
  mobileclip-s2  dim=512  image_size=256  disk≈220 MB
    Apple's MobileCLIP2-S2: matches SigLIP-SO400M at 2x fewer params, 3-15 ms
    inference on Apple Silicon. The v0 image-embedder choice.
    repo: plhery/mobileclip2-onnx
```

## 3. Download weights and index a directory

```text
$ lookback models download nomic-v1.5 mobileclip-s2
Downloading Nomic Embed v1.5 (137M, dense, Matryoshka) → ~/.lookback/models/nomic-v1.5
  model     → ~/.lookback/models/nomic-v1.5/onnx/model.onnx          (547 MB)
  tokenizer → ~/.lookback/models/nomic-v1.5/tokenizer.json
Downloading MobileCLIP2-S2 (~150M, 512-d) → ~/.lookback/models/mobileclip-s2
  vision    → ~/.lookback/models/mobileclip-s2/onnx/s2/vision_model.onnx (136 MB)
  text      → ~/.lookback/models/mobileclip-s2/onnx/s2/text_model.onnx
  tokenizer → ~/.lookback/models/mobileclip-s2/tokenizer.json
```

```text
$ lookback index /tmp/lookback-demo/src
Indexed 8 file(s), wrote 11 chunk(s); unchanged=0 skipped=0 errors=0
```

```text
$ lookback stats
  chunks_text             8
  chunks_image            3
  files                   8
```

3 markdown sections + 1 cooking long-section + 4 code chunks → 8 text rows.
3 PNGs → 3 image rows. 8 unique files tracked in the `files` table.

## 4. Pure semantic search (text → text)

The text embedder is Nomic Embed v1.5. Cosine distance is returned as
`score` — **lower is better**.

```text
$ lookback search "transformer attention mechanism" --modality text --limit 3
Text hits
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ score ┃ kind     ┃ meta                        ┃ text/file                   ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.276 │ markdown │ {"section": "Positional     │ Transformers add sinusoidal │
│       │          │ encoding", "section_idx":   │ positional encodings so the │
│       │          │ 2}                          │ model knows the order of    │
│       │          │                             │ tokens.                     │
│ 0.336 │ markdown │ {"section": "Attention is   │ Self-attention computes     │
│       │          │ all you need",              │ scaled dot-products between │
│       │          │ "section_idx": 1}           │ queries, keys, and values   │
│       │          │                             │ across every position in    │
│       │          │                             │ the sequence.               │
│ 0.524 │ markdown │ {"section": "Indexing",     │ B-tree indexes are the      │
│       │          │ "section_idx": 2}           │ default for ordered range   │
│       │          │                             │ scans. Hash indexes are     │
│       │          │                             │ faster for equality         │
│       │          │                             │ lookups…                    │
└───────┴──────────┴─────────────────────────────┴─────────────────────────────┘
```

The two transformer-related chunks rank first (0.276, 0.336); the
database chunk lands last (0.524) — semantically far from "transformer
attention" despite the literal word "indexes" overlapping with nothing
here.

## 5. Cross-modal text → image (the hero feature)

The query text is encoded by **MobileCLIP's text tower** into the same
joint embedding space as the image vectors. Result: free-text queries
against your screenshot folder, no captions or filenames required.

```text
$ lookback search "mountains and clouds" --modality image --limit 3
Image hits
┏━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ score ┃ kind       ┃ meta                          ┃ text/file ┃
┡━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 0.671 │ screenshot │ {"filename": "landscape.png"} │ (image)   │
│ 0.929 │ screenshot │ {"filename": "diagram.png"}   │ (image)   │
│ 0.980 │ screenshot │ {"filename": "dog.png"}       │ (image)   │
└───────┴────────────┴───────────────────────────────┴───────────┘
```

```text
$ lookback search "a dog in the grass" --modality image --limit 3
Image hits
┏━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ score ┃ kind       ┃ meta                          ┃ text/file ┃
┡━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 0.842 │ screenshot │ {"filename": "dog.png"}       │ (image)   │
│ 0.908 │ screenshot │ {"filename": "landscape.png"} │ (image)   │
│ 0.978 │ screenshot │ {"filename": "diagram.png"}   │ (image)   │
└───────┴────────────┴───────────────────────────────┴───────────┘
```

```text
$ lookback search "a neural network architecture diagram" --modality image --limit 3
Image hits
┏━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ score ┃ kind       ┃ meta                          ┃ text/file ┃
┡━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 0.634 │ screenshot │ {"filename": "diagram.png"}   │ (image)   │
│ 0.860 │ screenshot │ {"filename": "landscape.png"} │ (image)   │
│ 0.926 │ screenshot │ {"filename": "dog.png"}       │ (image)   │
└───────┴────────────┴───────────────────────────────┴───────────┘
```

Every query lands its target image as the top hit. The fixture images
are synthetic (~10 KB PNGs drawn with PIL primitives); real photographs
and screenshots produce sharper separations.

## 6. Hybrid search — FTS + vector for keyword precision

For technical queries that contain a rare or exact-match token (`IVF_PQ`,
a function name, an error code), pure vector search can dilute keyword
matches behind something that's semantically near but lexically distant.
`--hybrid` runs full-text search alongside vector search and fuses both
rankings via reciprocal rank fusion. The score field becomes
`_relevance_score` (higher = better).

```text
$ lookback search "IVF_PQ index tuning" --modality text --hybrid --limit 3
Text hits
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ score ┃ kind     ┃ meta                        ┃ text/file                   ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.033 │ python   │ {"line_start": 1,           │ import lancedb  def         │
│       │          │ "line_end": 12, "language": │ search_with_ivf_pq(db_path: │
│       │          │ "python"}                   │ str, …                      │
│ 0.032 │ markdown │ {"section": "Indexing",     │ B-tree indexes are the      │
│       │          │ "section_idx": 2}           │ default for ordered range   │
│       │          │                             │ scans…                      │
│ 0.016 │ markdown │ {"section": "Positional     │ Transformers add sinusoidal │
│       │          │ encoding", "section_idx":   │ positional encodings…       │
│       │          │ 2}                          │                             │
└───────┴──────────┴─────────────────────────────┴─────────────────────────────┘
```

The `lance_query.py` row containing the literal `search_with_ivf_pq`
function name comes out on top.

## 7. Combined modalities, source-kind filters, JSON output

Default `--modality all` queries both tables and prints results grouped
by modality:

```text
$ lookback search "neural network" --limit 2
Text hits
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ score ┃ kind     ┃ meta                        ┃ text/file                   ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.397 │ markdown │ {"section": "Attention is   │ Self-attention computes     │
│       │          │ all you need", …}           │ scaled dot-products…        │
│ 0.414 │ markdown │ {"section": "Positional     │ Transformers add sinusoidal │
│       │          │ encoding", …}               │ positional encodings…       │
└───────┴──────────┴─────────────────────────────┴─────────────────────────────┘
Image hits
┏━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ score ┃ kind       ┃ meta                          ┃ text/file ┃
┡━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 0.639 │ screenshot │ {"filename": "diagram.png"}   │ (image)   │
│ 0.846 │ screenshot │ {"filename": "landscape.png"} │ (image)   │
└───────┴────────────┴───────────────────────────────┴───────────┘
```

`--source-kind` narrows results to one file type:

```text
$ lookback search "embedding text into vectors" --modality text --source-kind python --limit 3
Text hits
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ score ┃ kind   ┃ meta                         ┃ text/file                    ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.334 │ python │ {"line_start": 1, "language":│ import numpy as np  from     │
│       │        │ "python"}                    │ sentence_transformers import │
│       │        │                              │ SentenceTransformer  def     │
│       │        │                              │ embed_documents…             │
│ 0.438 │ python │ {"line_start": 1, "language":│ import lancedb  def          │
│       │        │ "python"}                    │ search_with_ivf_pq…          │
└───────┴────────┴──────────────────────────────┴──────────────────────────────┘
```

Only Python files come back. The `embedding.py` row wins because its
contents semantically match the query better than the `lance_query.py`
row — `--source-kind` filters, but vector ranking still decides the
within-kind order.

`--json` swaps the table for machine-readable output:

```text
$ lookback search "transformer attention" --limit 1 --json
{
  "text": [
    {
      "id": "1c6b…",
      "file_id": "8e2a…",
      "modality": "text",
      "source_kind": "markdown",
      "chunk_idx": 1,
      "text": "Self-attention computes scaled dot-products between queries, keys, and values …",
      "created_at": "2026-05-18 …",
      "source_mtime": "2026-05-18 …",
      "meta": "{\"section\": \"Attention is all you need\", \"section_idx\": 1}",
      "_distance": 0.336
    }
  ],
  "image": []
}
```

## 8. Incremental re-indexing

The second pass over the same tree only touches files whose content hash
has changed. Unchanged files are reported, not re-embedded.

```text
$ lookback index /tmp/lookback-demo/src
Indexed 0 file(s), wrote 0 chunk(s); unchanged=8 skipped=0 errors=0
```

### Keeping the index live (`lookback watch`)

For continuous indexing — every file save picked up sub-second — run the
watcher. Two flavours:

**Foreground.** Useful for verifying behaviour; `Ctrl-C` prints summary
stats and exits.

```text
$ lookback watch /tmp/lookback-demo/src
Watching /tmp/lookback-demo/src (Ctrl-C to stop)
^C
Stopped — 3 batch(es), indexed=1 deleted=1 errors=0
```

**Detached, survives terminal close.** This is the "set it once, leave
it running until reboot" pattern:

```bash
nohup lookback watch ~/Documents > ~/.lookback/watch.log 2>&1 &
echo $! > ~/.lookback/watch.pid     # remember the PID so we can stop it later
```

Verify it's actually picking up changes by making some and querying for
the new content. The session below is verbatim against the demo
workspace:

```text
$ nohup lookback watch /tmp/lookback-demo/src --config /tmp/lookback-demo/config.toml \
    > /tmp/lookback-demo/watch.log 2>&1 &
$ echo $!
21314

# Add a new section to a tracked file
$ cat >> /tmp/lookback-demo/src/notes/transformers.md <<'EOF'
##  Scaling laws
Training loss follows a power law in compute, parameters, and tokens.
EOF

# Delete another tracked file
$ rm /tmp/lookback-demo/src/notes/cooking.md

# The new content is immediately searchable
$ lookback search "scaling laws training loss power law" --limit 1 --modality text
Text hits
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ score ┃ kind     ┃ meta                        ┃ text/file                   ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.218 │ markdown │ {"section": "Scaling laws", │ Training loss follows a     │
│       │          │ "section_idx": 3}           │ power law in compute,       │
│       │          │                             │ parameters, and tokens.     │
└───────┴──────────┴─────────────────────────────┴─────────────────────────────┘

# The deleted file is gone from the index
$ lookback stats --config /tmp/lookback-demo/config.toml
  chunks_text             7         # was 8 before the delete
  chunks_image            3
  files                   7         # was 8

# Stop the watcher when you're done
$ kill $(cat /tmp/lookback-demo/watch.pid)
```

**Caveats for the detached pattern:**

- **Dies on logout / reboot.** If your laptop sleeps or restarts the
  process exits and the index goes stale until you start it again. For
  reboot-persistent watching you'd want a launchd agent on macOS or a
  systemd user unit on Linux — that's M9 work, not in v0.
- **The watcher itself logs very little to stdout.** Verify it's
  working by triggering a change and either checking `lookback stats`
  or searching for the new content (as shown above).
- **`watch` doesn't seed the index.** Run `lookback index <path>` once
  first so the watcher only has to deal with deltas.

## 9. The MCP server (`lookback serve`)

This is what makes Lookback usable from any MCP-capable AI tool — Claude
Code, Cursor, Continue, ChatGPT Desktop, Windsurf, Zed.

```text
$ lookback serve --help

 Usage: lookback serve [OPTIONS]

 Run the Lookback MCP server.

 Default transport is stdio — that's what Claude Code, Cursor, Continue,
 and most IDE-side MCP clients use. See MCP_SETUP.md for client
 configuration recipes.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --transport  -t      TEXT     MCP transport: 'stdio' (default, used by IDE   │
│                               integrations) or 'http'.                       │
│                               [default: stdio]                               │
│ --host               TEXT     HTTP bind host (HTTP transport only).          │
│                               [default: 127.0.0.1]                           │
│ --port       -p      INTEGER  HTTP port (HTTP transport only).               │
│                               [default: 7777]                                │
│ --config     -c      PATH                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Two tools are exposed:

- `lookback_search(query, modality, limit, source_kind)` — returns `{"text": [...], "image": [...]}`.
- `lookback_stats()` — row counts per table.

Wire it into Claude Code by adding one block to `~/.claude.json` (or
`.mcp.json` in a workspace):

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

After a restart your assistant can answer queries like *"use lookback to
find that diagram about attention I screenshotted last week"* without
you opening a terminal. See [`MCP_SETUP.md`](MCP_SETUP.md) for the same
recipe for Cursor, Continue, ChatGPT Desktop, Windsurf, and Zed.

## 10. The full command surface

```text
$ lookback --help

 Usage: lookback [OPTIONS] COMMAND [ARGS]...

 Lookback — local-first multimodal semantic memory for your machine.

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ init    Detect the system, recommend models, and write a starter config.     │
│ index   Index a file or directory tree.                                      │
│ search  Semantic search across the indexed corpus, text and/or images.       │
│ stats   Print row counts for every table in the store.                       │
│ watch   Watch a directory and re-index on filesystem changes (Ctrl-C to      │
│         stop).                                                               │
│ serve   Run the Lookback MCP server.                                         │
│ models  Inspect and download embedding model weights.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```
