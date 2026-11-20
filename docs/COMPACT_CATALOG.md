# Compact catalog cache

Use `--catalog-precision int8` in the ONNX CLI or MCP stdio client command to compress cached tool embeddings with either pooled encoder graph. Float32 remains the default. Keep `catalog_int8.onnx` beside `rank_onnx.py`. Releases bundle it with both locked uv scripts. [ONNX setup](ONNX_INT8.md) · [MCP setup](MCP_SERVER.md)

```sh
uv run --locked --script examples/rank_onnx.py ./toolrank-int8 ./catalog.json \
  "Find my calendar meeting" --catalog-precision int8
```

| Benchmark | Float32 cache | INT8 cache |
| --- | ---: | ---: |
| ToolRet storage, 44,453 catalog vectors | 65.1 MiB | 16.4 MiB |
| ToolRet CPU scoring and top-10 p50 | 1.31 ms | 0.29 ms |
| ToolRet nDCG@10, 7,961 requests | 28.53% | 28.57% |
| ETOM nDCG@10, 1,549 answerable requests | 90.13% | 90.14% |
| ETOM no-match requests accepted at fixed threshold | 30/91 | 28/91 |

Four-thread Ryzen 9 5950X measurements exclude request encoding, catalog metadata, and inference runtime memory. ToolRet top-1 agrees on 95.3% of requests. Scoring ETOM’s 2,375 tools took 0.033 ms (float32) and 0.058 ms (INT8). Prefer float32 for small-catalog latency. Calibrate rejection per catalog because quantization changes individual scores.

Run from the repository root. For CPU, use `--extra cpu` and `--device cpu`. The reproducer pins benchmark inputs separately and normalizes embeddings once per catalog before comparing CPU scoring modes.

```sh
uv sync --locked --extra cu128 --group onnx --group research
uv run --no-sync hf download christianengel/mcp-toolrank-33m \
  --revision 08b109f86e88c43304732fcb45432dc67c294cdb --local-dir artifacts/toolrank
uv run --no-sync python -m examples.benchmark_catalog_index artifacts/toolrank \
  --fetch-evaluation --output results/catalog-cache-replay.json
```

[Full scores and timings](../results/catalog-cache.json)
