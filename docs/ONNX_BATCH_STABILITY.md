# Batch-stable ONNX pilot

Dynamic INT8 activation scales vary by batch. Encoder, tokenizer, inputs and runtime matched. Static calibration used 132 training texts from 44 fictional `data/train.json` tools. Frozen candidates rebuilt byte-for-byte.

| Graph | MiB | Short query p95, ms | ETOM hybrid nDCG@10 | Same ETOM top 10 after rebatching |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 126.96 | 4.85 | 92.69% | 100% |
| Dynamic INT8 | 32.52 | 2.54 | 92.70% | 4.39% |
| Static INT8 | 32.65 | 32.12 | 91.17% | 100% |
| Weight-only INT4 | 20.36 | 4.44 | 92.72% | 100% |

ETOM compared ID-vs-length batching (32 per batch, 2,375 tools, 1,549 answerable queries). Latency used 64 warmed development queries (four-thread Ryzen 9 5950X). API-only replay (24 tools) exactly matched metrics and deltas. Ranking excludes unsupported requests.

INT4 shrank 37.4%. ETOM difference +0.02 points (paired 95% interval −0.19 to +0.24). Three rotated dev rounds put INT4 with INT8 arithmetic at 3.95 ms p95 versus 2.59 ms dynamic. Both settings showed zero drift (eight dev pairs) and missed the 1.15× cap. Keep dynamic INT8.

## Reproduce

[Checkpoint](https://huggingface.co/christianengel/mcp-toolrank-33m/tree/a24a121266154006e2bd2e32a41576437c3455c2) · [ETOM setup](BASELINE_COMPARISON.md#reproduce)

```sh
uv sync --locked --extra cpu --group research --group onnx
fp32=/path/to/fp32-snapshot
int8=/path/to/int8-snapshot
pilot=.work/onnx-batch-pilot

uv run --no-sync python -m toolrank.export_batch_stable \
  --fp32 "$fp32" --int8 "$int8" --output "$pilot" --variant static-int8
uv run --no-sync --with onnx-ir==1.1.0 python -m toolrank.export_batch_stable \
  --fp32 "$fp32" --int8 "$int8" --output "$pilot" --variant weight-int4
uv run --no-sync python -m toolrank.onnx_batch_pilot \
  --fp32 "$fp32" --int8 "$int8" --candidates "$pilot" --output results/onnx_batch_pilot.json
uv run --no-sync python -m toolrank.probe_int4_execution \
  --int8 "$int8" --candidates "$pilot" --output results/int4_execution_dev.json
```

[Protocol](../toolrank/onnx_batch_protocol.json) · [Full results](../results/onnx_batch_pilot.json) · [API replay](../results/onnx_batch_api_replay.json) · [Development execution screen](../results/int4_execution_dev.json)
