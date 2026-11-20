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

Try published examples or paste MCP `tools/list` responses, result objects, or arrays (up to 128 tools). A browser worker runs the 34 MB INT8 model, caching tool embeddings locally. Requests and catalogs stay in your browser. First use downloads the model, tokenizer, and WebAssembly runtime.

After querying, compare Dense and Hybrid top-five results without re-embedding. Dense ranks by embedding cosine. Hybrid adds TF-IDF and name-weighted BM25. Displayed scores remain embedding cosine. Benchmark examples label matches and no-matches. Unlabeled requests can score high without an applicable tool.

**Evaluate the API holdout** compares Hit@1 and nDCG@10 locally over 72 answerable requests and 24 tools. Metrics exclude 24 no-match requests. Rejection remains unmeasured.

## Run locally

Run in the repository’s `demo` directory.

```sh
npm ci
npm run dev
```

`npm run build` generates the static Space bundle.

[Model](https://huggingface.co/christianengel/mcp-toolrank-33m) · [API holdout](https://huggingface.co/datasets/christianengel/toolrank-api-holdout) · [Source and integration guides](https://github.com/protortyp/mcp-toolrank)
