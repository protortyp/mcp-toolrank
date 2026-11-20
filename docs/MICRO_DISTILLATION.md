# 17M tool retrieval pilot

MIT-tagged three-layer BGE Micro trained three RTX 3090 seeds (one epoch, 512 tokens), sharing 5,098 audited requests, 263 tool descriptions, and masked teacher-mined negatives.

## Development

Family-macro hybrid nDCG@10 (872 requests, 34 tools, nine disjoint families).

| Encoder | nDCG@10 |
| --- | ---: |
| Untuned BGE Micro | 92.67% |
| Untuned BGE Small | 93.55% |
| ToolRank 33M teacher | 94.85% |
| Micro, labels only, three seeds | 94.42–94.51% |
| Micro, distillation, three seeds | 94.45–94.46% |

Distillation intervals included zero. Retrained scores matched within 0.003 points. Weight hashes differed. [Replay](../results/micro_distillation_replay.json)

Label-only INT8 measured 17.48 vs 34.09 MB. Four-thread Ryzen 9 5950X p95 fell 4.21→1.15 ms short, 66.14→24.08 ms long.

## Held-out results

Frozen gates evaluated label-only seed 11 on answerable requests.

| Holdout | Micro nDCG@10 | ToolRank nDCG@10 | Difference, points (95% interval) | Gate |
| --- | ---: | ---: | ---: | --- |
| Independent API | 92.00% | 91.57% | +0.43 [-1.54, +2.99] | Pass |
| ETOM compact | 91.86% | 92.70% | -0.84 [-1.30, -0.37] | Fail |
| ETOM MCP JSON | 91.86% | 91.89% | -0.03 [-0.61, +0.55] | Pass |
| ETOM OpenAI JSON | 91.44% | 91.80% | -0.35 [-0.95, +0.24] | Pass |
| ToolRet | 35.06% | 37.00% | −1.94 [−3.41, −0.58] | Fail |

**Keep ToolRank 33M.** Micro missed preregistered lower-bound gates on compact ETOM and ToolRet. In one matched four-thread CPU run, 44,453-tool indexing took 821.34 s for Micro and 2,997.76 s for ToolRank. Hit@5 fell 2.26 points (35-source paired 95% interval −4.30 to −0.37), missing the −1.00-point floor. [Overlap audit](../results/micro_holdout_overlap.json)

RTX 3090 FP32 ToolRet delta −2.13 hybrid nDCG@10 points (35-source paired 95% interval −3.59 to −0.84). Indexing took 36.56 vs 98.71 s. [GPU results](../results/micro_toolret_gpu.json)

## Reproduce

Archived replay caches by graph, ordered text and batching (`--catalog-cache DIR`).

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
.work/venv-micro-cu/bin/python -m toolrank.profile_micro_gpu_toolret \
  --teacher "$teacher" --candidate /path/to/archived-original-models/labels-11 \
  --training-record results/micro_distillation_training.json --output "$replay/toolret-gpu.json"
```

[Training](../results/micro_distillation_training.json) · [Development](../results/micro_distillation_development.json) · [CPU label-only](../results/micro_labels_cpu.json) · [CPU distilled](../results/micro_distillation_cpu.json) · [Frozen gate](../toolrank/micro_holdout_protocol.json) · [Holdout](../results/micro_holdout.json)

Development audit found 242 shared top-one misses. Training pairs included clear negatives and plausible alternatives. Review query-specific labels before changing the mask. [Training data audit](MICRO_TRAINING_AUDIT.md)
