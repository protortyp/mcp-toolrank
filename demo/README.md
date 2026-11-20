---
title: ToolRank MCP Selector
emoji: 🧭
colorFrom: green
colorTo: gray
sdk: static
app_build_command: npm ci && npm run build
app_file: dist/index.html
license: apache-2.0
models:
- christianengel/mcp-toolrank-33m
---

# ToolRank MCP Selector

Try the sample catalog or paste an MCP `tools/list` JSON-RPC response, result object, or array of tool definitions. Enter a request to rank the top five tools by cosine similarity. Catalogs can contain up to 128 tools. High scores can point to unsuitable tools when the catalog lacks a relevant match.

The first run downloads the 34 MB INT8 ONNX model, a WebAssembly runtime, and a tokenizer. A browser worker processes requests and catalog contents on your device and caches catalog embeddings in memory.

## Run locally

In `demo`, install dependencies and start Vite.

```sh
npm ci
npm run dev
```

Build the Space’s static files with `npm run build`.

[Model and evaluation](https://huggingface.co/christianengel/mcp-toolrank-33m) · [Source and Python integration](https://github.com/protortyp/mcp-toolrank)
