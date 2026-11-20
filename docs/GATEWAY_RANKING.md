# BM25 and hybrid gateway ranking

Hybrid remains default after leading ETOM. BM25 suits fast starts and shared vocabulary. The BM25 wheel discovered clock and Git tools and called the clock offline with an empty model cache.

| Catalog | Mode | nDCG@10 (%) | All positives in top ten | Cold first search | Warm p50 | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, 93 tools | BM25 | 85.77 | 18/20 | 0.27 s | 0.34 ms | 82 MiB |
| Mixed, 93 tools | Hybrid | 86.70 | 17/20 | 0.77 s | 4.07 ms | 187 MiB |
| ETOM, 2,375 tools | BM25 | 88.68 | 1,427/1,549 | 0.32 s | 0.94 ms | 126 MiB |
| ETOM, 2,375 tools | Hybrid | 92.70 | 1,465/1,549 | 9.42 s | 6.04 ms | 287 MiB |

Three fresh processes per mode timed `Gateway.search` with matched catalogs, labels, schemas, tie breaks, and shuffled queries. Cold uses warm files and empty embeddings. Timing covers model load, indexing, and ranking. Downloads, MCP transport, and fixture loading excluded. Process RSS measured on shared Ryzen 9 5950X.

Mixed uses pinned NWS OpenAPI and six API fixtures. ETOM uses registry-filtered official level 1/2/5 labels with flattened level 2 equivalence groups. BM25 drops zero scores. Both suggested tools for all 8 mixed and 91 ETOM unsupported requests. JSON records checkpoint fingerprints, seeds, and warm p95.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/gateway_profile.py \
  --checkpoint /path/to/toolrank-int8 --output /tmp/mixed-gateway.json
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/gateway_profile.py \
  --checkpoint /path/to/toolrank-int8 --etom .work/next/etom --output /tmp/etom-gateway.json
```

[Mixed catalog](../results/gateway_ranking_profile.json) · [ETOM](../results/gateway_etom_profile.json) · [Baseline](BASELINE_COMPARISON.md) · [Offline smoke](../results/gateway_bm25_smoke.json)
