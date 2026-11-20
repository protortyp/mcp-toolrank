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

Try a sample or select **Tool catalog** → **Import file**. Paste JSON or import MCP tool lists and single-file OpenAPI 3.0/3.1 JSON or YAML. Import files can be up to 1 MB, with at most 128 tools or operations. OpenAPI import reads path operations, resolves same-file path item references, and rejects external ones. Import creates search entries only.

First use downloads the 34 MB INT8 model and WebAssembly runtime. A browser worker handles embedding and lexical ranking. Requests and catalogs stay in your browser.

Switch between Dense and Hybrid without re-encoding queries. Scores show embedding cosine. Hybrid ranking adds TF-IDF and BM25. Examples label matches and no-matches.

**Evaluate the API holdout** reports Hit@1 and nDCG@10 across 72 answerable requests and 24 tools. It skips 24 no-matches. Rejection remains unmeasured.

## Run locally

Run from `demo`.

```sh
npm ci
npm run dev
```

Build the static bundle with `npm run build`. Test the parser with `npm test`.

[Model](https://huggingface.co/christianengel/mcp-toolrank-33m) · [API holdout](https://huggingface.co/datasets/christianengel/toolrank-api-holdout) · [Source](https://github.com/protortyp/mcp-toolrank)
