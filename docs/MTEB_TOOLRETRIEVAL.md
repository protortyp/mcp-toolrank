# MTEB ToolRetrieval replay

The official MTEB 2.27.18 ToolRetrieval task covers 35 subsets and shares a corpus of 44,453 tools. MTEB pins the dataset revision. The runner pins model revisions and saves all metrics, including per-subset results. Published artifacts contain aggregate metrics and revision metadata. Benchmark inputs retain tool descriptions and requests.

| Model | nDCG@10 |
| --- | ---: |
| Released ToolRank 33M | 22.62% |
| BGE Small | 22.49% |

MTEB weights subsets equally. The earlier ToolRet report weights queries equally. The ToolRank repeat matched all 1,015 per-subset retrieval metrics.

From the repository root, run either model with the same task and encoding settings. The first run downloads pinned files and caches embeddings for reuse across subsets.

```sh
uv run --locked --extra cu128 --group mteb python -m toolrank.replay_mteb \
  --model toolrank --output artifacts/mteb/toolrank.json
uv run --locked --extra cu128 --group mteb python -m toolrank.replay_mteb \
  --model bge-small --output artifacts/mteb/bge-small.json
```

For CPU, replace `--extra cu128` with `--extra cpu` and append `--device cpu`.

[ToolRank result](../results/mteb/toolrank-ToolRetrieval.json) · [BGE result](../results/mteb/bge-small-ToolRetrieval.json) · [MTEB task dataset](https://huggingface.co/datasets/mteb/ToolRetrieval)
