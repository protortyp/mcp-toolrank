# Evaluate ToolRank on your tool catalog

Compare INT8 and FP32 CPU rankings on the same requests using local files.

## Prepare the inputs

Save `tools/list` as `catalog.json`. Use the full JSON-RPC response, its result object, or the tools array. Give each tool a unique name. In `labels.json`, list representative requests and all acceptable tool names in `relevant`. Use `[]` for unsupported requests.

```json
[
  {"query": "Find tomorrow's meetings", "relevant": ["search_calendar"]},
  {"query": "Translate this paragraph into Danish", "relevant": []}
]
```

Use the official model download command and add `'onnx/embedding.onnx'` to `--include` to fetch the FP32 graph alongside the INT8 graph, tokenizer, and SentenceTransformer configs. [CPU download instructions](ONNX_INT8.md)

## Run the benchmark

```sh
uv run --locked examples/rank_onnx.py artifacts/toolrank-int8 catalog.json \
  --evaluate labels.json --limit 5 --compare-model-file embedding.onnx
```

The JSON report gives Hit@k, Recall@k, and binary nDCG@k for answerable requests, independent of the score threshold. Median and p95 CPU query latency after warm-up cover tokenization and ranking. Graph loading and catalog embedding occur before timing.

Pass `--min-score` to measure candidate return rates for answerable and unsupported requests. By default, both graphs return candidates for every request when the catalog is nonempty. Choose thresholds per graph and catalog on a separate development set, then measure results on untouched test requests.
