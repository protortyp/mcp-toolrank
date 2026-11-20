# Release review

**Recommend releasing the 33.36M checkpoint for English tool shortlisting, subject to final approval.** The default selector ranks compact descriptions. On that representation, the model improved over both its own base and the same-size specialist on ETOM. The gain over its base reproduced with independent NumPy ranking and all three conservative training seeds. ToolRet scores are close to both same-size comparators. The release claim is limited to static retrieval on the measured catalog and representation.

| Model | Parameters | ETOM plain nDCG@10 | ETOM MCP nDCG@10 | ETOM OpenAI nDCG@10 | ToolRet nDCG@10 | CPU p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE small | 33.36M | 88.95% | 87.94% | 87.69% | 28.81% | 13.57 ms |
| ToolScout | 33.36M | 81.42% | 80.73% | 80.14% | 28.66% | 12.70 ms |
| Tool-Embed-0.6B | 595.78M | 93.06% | 93.00% | 93.60% | 39.43% | 196.06 ms |
| Fine-tuned BGE micro | 17.39M | 89.03% | 87.53% | 87.40% | 27.94% | 3.52 ms |
| **Release candidate** | **33.36M** | **90.13%** | **87.98%** | **87.67%** | **28.54%** | **12.71 ms** |

The compact-description gain over BGE was 1.18 points of nDCG@10, with a paired 95% interval of 0.53 to 1.85 points. Hit@1 improved by 2.26 points, with a paired interval of 0.94 to 3.60 points. The three fixed conservative seeds scored 90.13%, 90.24%, and 90.36% nDCG@10 with independent NumPy evaluation. Native MCP and OpenAI JSON measurements were close to the BGE baseline. On the original 12-tool authored test, the release candidate had 83.33% Recall@1 with compact descriptions. The 17.39M branch is a speed option for further research. Its plain ETOM gain over its own base was within the paired interval of −0.46 to +0.63 points, and it had weaker abstention than the recommended checkpoint. [Paired comparison](../results/etom/plain-paired.json) · [Replica 23](../results/etom/conservative-23-plain-replay.json) · [Replica 37](../results/etom/conservative-37-plain-replay.json) · [Micro comparison](../results/etom/micro-plain-paired.json)

On ETOM with compact descriptions, nDCG@10 was 96.14% for single-tool judgments and 84.02% for requests with equivalent tools. The threshold calibrated on authored development queries accepted 30 of 91 infeasible ETOM requests and retained 1,547 of 1,549 answerable requests. Native MCP JSON accepted 57 of 91 infeasible requests with its own independently calibrated threshold. Treat similarity as a ranking signal and calibrate rejection separately for the target catalog. [Plain graded results](../results/etom/candidate-plain.json) · [MCP graded results](../results/etom/candidate-mcp.json)

The exact overlap audit found zero shared normalized tools, requests, or identified repositories between the training and development splits and ETOM. A GitHub ancestry check covered 49 training and development repositories and found no matching known held-out provider. Repository aliases or unknown model pretraining exposure remain possible. [Split audit](../results/mcp-split-independence.json) · [Fork audit](../results/mcp-fork-audit.json)

The candidate is packaged locally as a standard SentenceTransformer with a CPU ONNX graph and 39 upstream source notices, along with the base model and data notices. Apache-2.0 is proposed for the project code and candidate checkpoint, with the original MIT and Apache-2.0 notices retained. The review includes aggregate benchmark outputs and pinned download instructions. Raw upstream benchmark and training snapshots remain local. External spend was $0. [Model card](MODERN_MODEL_CARD.md) · [Technical article](MODERN_ARTICLE.md) · [Spending ledger](SPEND.md)

## Runtime on the local Ryzen 9 5950X and RTX 3090

| Model | CPU p95 | CPU batch-16 queries/s | CPU peak RSS | RTX p50 | RTX p95 | RTX batch-16 queries/s | RTX peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE small | 16.34 ms | 198 | 544 MiB | 8.61 ms | 11.22 ms | 1645 | 143 MiB |
| ToolScout | 14.78 ms | 210 | 544 MiB | 6.94 ms | 8.54 ms | 2060 | 143 MiB |
| Tool-Embed-0.6B | 222.85 ms | 10 | 3824 MiB | 25.26 ms | 28.68 ms | 306 | 2318 MiB |
| Fine-tuned BGE micro | 4.45 ms | 894 | 499 MiB | 7.84 ms | 10.57 ms | 2217 | 80 MiB |
| Release candidate | 14.96 ms | 215 | 567 MiB | 7.09 ms | 8.39 ms | 2164 | 143 MiB |

Latency includes query tokenization and encoding. Batch throughput uses batch 16. The GPU was shared with other local workloads. [Runtime records](../results/modern-runtime)
