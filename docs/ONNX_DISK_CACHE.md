# Reuse ONNX catalog vectors on disk

Pass `--catalog-cache DIR` to cache vectors on first launch and reuse them in the CPU ranker or MCP stdio. Reuse requires matching tool names, descriptions, checkpoint, precision, runtime, and thread count.

```sh
uv run --locked --script examples/rank_onnx.py artifacts/toolrank-int8 catalog.json \
  "Find museum exhibits open today" --hybrid --catalog-precision int8 \
  --catalog-cache artifacts/toolrank-catalog-cache
```

Use the same flag in the [MCP stdio command](MCP_SERVER.md). Float32 and INT8 vectors work. Schemas and metadata always come from the current catalog. Hybrid rebuilds its lexical index, and every query is encoded.

ToolRank rebuilds missing or malformed entries. Multiple entries can coexist, and old `.npz` files are removable. Four-thread CLI timing includes startup, ranking, and catalog encoding when needed. On 263 tools, median fell 6.03→0.33 seconds across three runs with identical outputs. [Timing runs](../results/onnx-disk-cache.json) · [ONNX setup](ONNX_INT8.md)
