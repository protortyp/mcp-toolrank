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

Use a sample or **Tool catalog** to import MCP or OpenAPI JSON or YAML (128 tools or operations, 1 MB). Local OpenAPI path references work. Supplied MCP schemas appear in results.

BM25 starts model-free and returns no matches without term overlap. Dense, hybrid and benchmarking load a 34 MB encoder and WebAssembly. Results show timings, shortlist bytes, and known no-matches. Custom applicability stays unverified.

Run from `demo`.

```sh
npm ci
npm run dev
```

Run `npm test` and `npm run build`. The benchmark reports Hit@1 and nDCG@10 on 72 answerable requests and 24 tools. [Model](https://huggingface.co/christianengel/mcp-toolrank-33m) · [API holdout](https://huggingface.co/datasets/christianengel/toolrank-api-holdout)
