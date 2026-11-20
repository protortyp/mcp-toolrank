# Cold hybrid indexing pilot

Matched INT8, lexical weights, catalogs, and top-ten output across strategies. The 64-tool shortlist falls back to full hybrid below ten lexical matches.

## Compact ETOM

| Strategy | First search, s | Later p95, ms | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.36 | 2.10 | 88.68% |
| Full hybrid | 10.52 | 21.39 | 92.70% |
| Lexical shortlist | 0.94 | 36.37 | 92.45–92.54% |
| Length-bucketed hybrid | 8.38 | 9.80 | 92.54% |

Medians across three fresh Ryzen 9 5950X runs without disk cache. First search includes imports and setup. Later p95 includes lazy embedding. Shared-machine timings varied. Replay excludes controller memory from RSS.

The shortlist embedded 64 of 2,375 descriptions initially and 2,374 overall. It met the cold target. Paired nDCG@10 lower bounds stayed within 0.51 points of full hybrid. Later p95 exceeded the 1.25× full-hybrid limit. Length bucketing missed 1.5× cold speedup and preserved 4.2% of ordered top tens. INT8 batching can alter rankings.

On the 24-tool API holdout, full hybrid scored 91.57% nDCG@10 and shortlisting 91.57–92.09%. All modes suggested tools for every unsupported query across both holdouts. Keep both experimental.

Replay matched quality exactly. Shortlisting traded 12.92→1.49 s cold for 24.16→45.84 ms later p95.

```sh
uv sync --locked --extra cpu --group research --group onnx
uv sync --project cpu --locked --extra gateway --no-editable
PYTHONPATH=. uv run --no-sync python tests/mixed_catalog/shortlist_pilot.py \
  --checkpoint /path/to/pinned-model-snapshot --etom /path/to/pinned-etom \
  --output results/shortlist_hybrid_replay.json
```

[ETOM setup](BASELINE_COMPARISON.md#reproduce) · [Frozen protocol](../tests/mixed_catalog/shortlist_protocol.json) · [First run](../results/shortlist_hybrid_pilot.json) · [Replay](../results/shortlist_hybrid_replay.json)
