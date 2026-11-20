# 17M tool retrieval pilot

Three-layer MIT-tagged BGE Micro used 5,098 audited requests and 263 tool descriptions. Shared teacher-mined negatives used provider/name uncertainty masks. Distillation added ToolRank 33M scores to label loss. RTX 3090, three one-epoch seeds, 512-token limit.

## Development

Family-macro hybrid nDCG@10 across 872 requests, 34 tools, nine disjoint repository families.

| Encoder | nDCG@10 |
| --- | ---: |
| Untuned BGE Micro | 92.67% |
| Untuned BGE Small | 93.55% |
| ToolRank 33M teacher | 94.85% |
| Micro, labels only, three seeds | 94.42–94.51% |
| Micro, distillation, three seeds | 94.45–94.46% |

Distillation shifted −0.04 to +0.03 points per seed. All paired intervals included zero. Both arms had 17.39M parameters. Retraining reproduced development within 0.003 points with different weight hashes. [Replay](../results/micro_distillation_replay.json)

INT8 label-only seed 11 versus ToolRank measured 17.48 vs 34.09 MB and 94.43% vs 94.65% macro development nDCG@10. Four-thread Ryzen 9 5950X p95 was 1.15 vs 4.21 ms short and 24.08 vs 66.14 ms long (64 interleaved calls per graph/workload).

## Held-out results

Only label-only seed 11 advanced. Protocol froze graph hash, matched hybrid scorer, size/latency limits, and per-catalog noninferiority before holdouts. Ranking excludes unsupported requests.

| Holdout | Micro nDCG@10 | ToolRank nDCG@10 | Difference, points (95% interval) | Gate |
| --- | ---: | ---: | ---: | --- |
| Independent API | 92.00% | 91.57% | +0.43 [-1.54, +2.99] | Pass |
| ETOM compact | 91.86% | 92.70% | -0.84 [-1.30, -0.37] | Fail |
| ETOM MCP JSON | 91.86% | 91.89% | -0.03 [-0.61, +0.55] | Pass |
| ETOM OpenAI JSON | 91.44% | 91.80% | -0.35 [-0.95, +0.24] | Pass |
| ToolRet | — | — | Pending | Running |

**Keep ToolRank 33M.** Compact ETOM failed the −1.00-point lower-bound gate. ToolRet CPU evaluation is running. Exact overlap checks found no training queries or tool texts in the holdouts. [Overlap audit](../results/micro_holdout_overlap.json)

## Reproduce

Held-out replay uses the exact archived local checkpoint.

[Data intake](MODERN_MCP.md) · [Base](https://huggingface.co/TaylorAI/bge-micro-v2/tree/3edf6d7de0faa426b09780416fe61009f26ae589) · [Teacher](https://huggingface.co/christianengel/mcp-toolrank-33m/tree/a24a121266154006e2bd2e32a41576437c3455c2)

```sh
UV_PROJECT_ENVIRONMENT=.work/venv-micro-cu uv sync --locked --extra cu128 --group research
uv sync --locked --extra cpu --group research --group onnx
student=.work/models/TaylorAI/bge-micro-v2
teacher=/path/to/toolrank-fp32
int8=/path/to/toolrank-int8
corpus=.work/next/modern-expanded
replay=.work/micro-distillation-replay
micro=$replay/models

.work/venv-micro-cu/bin/python -m toolrank.train_micro_distillation \
  --student "$student" --teacher "$teacher" --train "$corpus/train.json" \
  --output "$micro" --record "$replay/training.json"
.work/venv-micro-cu/bin/python -m toolrank.evaluate_micro_distillation \
  --student "$student" --teacher "$teacher" --dev "$corpus/dev.json" --models "$micro" \
  --training-record "$replay/training.json" --device cuda \
  --output "$replay/development.json"
uv run --no-sync python -m toolrank.profile_micro_distillation --arm labels \
  --int8 "$int8" --dev "$corpus/dev.json" --models "$micro" \
  --training-record "$replay/training.json" --output "$replay/labels-cpu.json"
uv run --no-sync python -m toolrank.evaluate_micro_holdout \
  --int8 "$int8" --models /path/to/archived-original-models \
  --cpu-profile results/micro_labels_cpu.json --output "$replay/holdout.json"
```

[Training](../results/micro_distillation_training.json) · [Development](../results/micro_distillation_development.json) · [CPU label-only](../results/micro_labels_cpu.json) · [CPU distilled](../results/micro_distillation_cpu.json) · [Frozen gate](../toolrank/micro_holdout_protocol.json) · [Holdout](../results/micro_holdout.json)
