# Matched retrieval baseline

**Keep the bundled weights. Skip retraining.** Fine-tuning reliably helped on compact ETOM. Tool-Embed-0.6B leads on ETOM and ToolRet.

Scores are nDCG@10 percentages.

| Catalog | Tools | BM25 | BGE hybrid | ToolRank hybrid | Difference, points (95% interval) | Tool-Embed dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETOM compact | 2,375 | 88.68 | 91.99 | 92.69 | +0.70 [+0.41, +1.00] | 93.06 |
| ETOM MCP JSON | 2,375 | 89.38 | 92.20 | 92.18 | −0.03 [−0.39, +0.34] | 93.00 |
| ETOM OpenAI JSON | 2,375 | 88.57 | 92.04 | 92.19 | +0.16 [−0.20, +0.51] | 93.60 |
| ToolRet | 44,453 | 32.50 | 37.10 | 37.19 | +0.10 [−1.15, +1.08] | 39.43 |
| API holdout | 24 | 88.36 | 90.70 | 92.09 | +1.39 [0.00, +3.14] | 88.87 |

ToolRet dense scores were 28.81 for BGE and 28.54 for ToolRank. Both hybrids hit every API positive at five. [All ten systems, Recall, Hit, completeness and intervals](../results/baseline_comparison.json)

All runs shared catalogs, queries, labels and stable tool-ID tie breaks. Dense runs used FP32 and 512 tokens. Development selected cosine + 0.40 TF-IDF + 0.15 BM25 for both hybrids. BM25 boosts compact and MCP names. Nested OpenAI names lack this boost. ETOM completeness counts equivalents individually. Paired bootstrap resampled providers or ToolRet sources (3,000 samples, seed 20261120). Ranking excludes unsupported requests.

ToolRet embedding on the shared RTX 3090 took 69 seconds each for BGE and ToolRank and 699 for Tool-Embed. The CPU profile covers load, indexing, first query, warm p50/p95, batch throughput, and resident memory. [CPU profile](../results/baseline_runtime.json)

## No-tool rejection

Cutoffs used six development negatives. ToolRank cosine passed 3/91 unsupported MCP requests and 10/24 API requests, retaining 1,530/1,549 and 71/72 supported requests. The API margin passed 2/24 unsupported and 34/72 supported. Calibrate locally and separate ranking from acceptance. [Risk and transfer](../results/abstention_comparison.json) · [Calibration](ONNX_CALIBRATION.md)

## Reproduce

```sh
uv sync --locked --extra cu128 --group research
uv run --no-sync python -m toolrank.fetch_baseline --toolrank artifacts/toolrank --toolret
uv run --no-sync python -m toolrank.baseline_comparison --device cuda \
  --toolrank artifacts/toolrank --output results/baseline_comparison.json
uv run --no-sync python -m toolrank.abstention_comparison \
  --toolrank artifacts/toolrank --output results/abstention_comparison.json
```

JSON records hashes, revisions, hardware, counts and timings. Train and test stay separate. [Aggregate results](../results/baseline_comparison.json)
