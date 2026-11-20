# ToolRank on CPU with INT8 ONNX

Opt-in INT8 ONNX returns normalized 384-dimensional embeddings with the same 512-token limit. SentenceTransformer weights and FP32 stay the defaults.

## Results

| Metric | FP32 ONNX | INT8 ONNX |
| --- | ---: | ---: |
| Graph size | 133.1 MB | 34.1 MB |
| Short query p50 | 6.11 ms | 3.53 ms |
| Long query p50 | 62.29 ms | 41.06 ms |
| ETOM compact nDCG@10 | 90.13% | 89.83% |
| ETOM compact Hit@1 | 84.51% | 83.99% |
| Fresh API holdout nDCG@10 | 92.15% | 90.43% |
| Fresh API holdout Hit@1 | 59 / 72 | 56 / 72 |

Timing includes tokenization and one embedding per query on a four-thread Ryzen 9 5950X, with 64 warmed, interleaved samples per graph and workload. The frozen holdout covers 96 requests across six providers, including 24 without a matching tool. Configuration selection excluded this holdout. We calibrated each graph on the original development split. INT8 accepted fewer answerable and unanswerable requests. Calibrate rejection on your catalog.

## Run on CPU

The standalone example requires no PyTorch. Pass an MCP `tools/list` result or tool array.

```sh
uvx --from huggingface-hub==0.36.2 hf download christianengel/mcp-toolrank-33m \
  --revision 51a961aa33a2e6dfecc6845df77203778787ec4a \
  --include 'onnx/embedding-int8.onnx' 'tokenizer.json' \
  'sentence_bert_config.json' 'config_sentence_transformers.json' \
  --local-dir artifacts/toolrank-int8
uv run --locked examples/rank_onnx.py artifacts/toolrank-int8 catalog.json \
  "Find tomorrow's meetings" --limit 3
```

## Reproduce

Use the full checkpoint and [ETOM mirror](MODERN_MCP.md). Export applies dynamic, per-channel QInt8 to MatMul, Gemm, and Gather.

```sh
uv sync --locked --extra cpu --group research --group onnx
uv run --no-sync python -m toolrank.quantize_onnx artifacts/toolrank/onnx/embedding.onnx artifacts/toolrank/onnx/embedding-int8.onnx
uv run --no-sync python -m toolrank.profile_onnx artifacts/toolrank --output artifacts/profile.json
uv run --no-sync python -m toolrank.validate_onnx artifacts/toolrank --output artifacts/validation.json
```

[Timing replay](../results/onnx-int8/cpu-replay.json) · [Validation](../results/onnx-int8/validation.json) · [Frozen API cases](../data/onnx_int8_holdout.json) · [License](../LICENSE)
