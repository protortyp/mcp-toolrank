# ONNX CLI and MCP hybrid ranking

Pass `--hybrid` to the ONNX CLI or MCP client command for `search_mcp.py`. The Python selector’s fixed weights (0.4 TF-IDF, 0.15 name-weighted BM25) augment embedding cosine. Dense-only is default.

Keep `catalog_lexical.py` beside `rank_onnx.py`. Download encoder and tokenizer per ONNX setup. Locked scripts score with NumPy and ONNX Runtime, without PyTorch or scikit-learn.

```sh
uv run --locked --script examples/rank_onnx.py ./toolrank-int8 ./catalog.json \
  "Find my calendar meeting" --hybrid --catalog-precision int8
```

Both pooled graphs and catalog precisions work. Returned `score` and `min_score` use embedding cosine. INT8 shrinks dense cache vectors. The sparse index adds RAM. Calibrate rejection per catalog.

## Validation

With shared embeddings, NumPy and package scorers matched ordered top 10 lists on all 7,961 ToolRet, 3,280 ETOM, and 96 API requests, including no-match queries. INT8 ONNX encoder nDCG@10 on 72 answerable API requests rose from 89.92% dense-only to 91.57% hybrid at both cache precisions.

```sh
uv run --locked --extra cu128 --group onnx --group research python -m examples.benchmark_onnx_hybrid \
  /path/to/model-snapshot --fetch-evaluation --output /tmp/onnx-hybrid.json
```

[Raw scores](../results/onnx-hybrid.json) · [ONNX model setup](ONNX_INT8.md) · [MCP stdio](MCP_SERVER.md) · [Python retrieval benchmarks](BM25.md)
