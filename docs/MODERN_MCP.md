# Real MCP corpus and evaluation

This experiment keeps the original project goal and uses the first pilot as a reference. The candidate starts from the same 33.36M BGE checkpoint with its 512-token serving limit. Experiments are local and the external ledger remains at $0.

## Frozen evaluation

ETOM release `790f28727f567b88918420d163c524d46644a694` provides real tool names, descriptions, and input schemas. The local evaluator ranks the full ordered catalog of 2,375 tools from 491 servers. It scores 781 single-tool requests, 768 requests with equivalent tools, and retains 91 infeasible requests for abstention. Five Level 2 requests were removed in their entirety because the release references one or more IDs missing from its registry. The exact input hashes and counts are recorded. ETOM provides overlapping groups of interchangeable choices for Level 2. The evaluator takes their union, uses Hit@K to measure whether a valid option appears, and separately reports recall and nDCG over all judged options. These are static retrieval scores. [ETOM paper](https://aclanthology.org/2026.findings-eacl.75/) · [Input audit](../results/etom-intake.json)

The following retrieval measurements use native MCP JSON and up to 512 tokens. Positive metrics are means over Levels 1 and 2. Infeasible queries have no positive judgment. Their rejection will require a development-calibrated threshold.

| Model | Recall@1 | Recall@5 | Recall@10 | nDCG@10 | Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE small | 76.95% | 91.75% | 94.44% | 87.94% | 80.89% |
| ToolScout | 67.69% | 85.82% | 90.01% | 80.73% | 70.95% |
| Tool-Embed-0.6B | 84.72% | 94.79% | 96.92% | 93.00% | 88.90% |

The ETOM release is used locally and its data is excluded from distribution. The MCP-Atlas feasibility check found that an independently captured catalog matched just 56 of 220 enabled names and 30 of 106 recorded target names after direct normalized matching. Without exact versioned descriptions, Atlas is reserved as a candidate evaluation rather than scored with guessed documents. Its provider names still participate in the training-side exclusion check. [ETOM baseline outputs](../results/etom) · [MCP-Atlas release](https://huggingface.co/datasets/ScaleAI/MCP-Atlas)

## Training data

The source intake reads only the Toucan request, declared positively labeled names, static definitions, and upstream repository metadata. It matches each named positive to the same file as its whitespace-normalized description in a pinned source archive. Each included source's MIT license file must match its recorded Git blob. This pass discards incomplete label rows and filters common contact details, URLs, and local path patterns. No trajectory text, arguments or tool outputs enter the dataset. Equivalent current source wording backs the description, while the original extraction commit and the full schema provenance remain unresolved. The model trains on names and matched descriptions.

Across six pinned shards, 74,115 rows yielded 5,970 unique eligible requests. The family split holds back 872 requests and 34 tools for development. Training uses 5,098 requests and 263 tools. The 45 repository groups are separated by normalized repository names and shared exact tool definitions. Registry reservations match source slugs and normalized provider names, including common MCP and server suffixes. Unknown forks and semantically equivalent wrappers remain a limitation. A character-gram check found zero development requests above 0.9 similarity to training. An eight-row manual spot check found plausible labels and one apparent capability-to-request gap around saving generated images. This does not establish annotation completeness across the corpus. [Source checks](../results/mcp-expanded-staging.json) · [Pinned licenses](../results/mcp-source-licenses-expanded.json) · [Split](../results/mcp-modern-split.json)

The first model run uses a multi-positive contrastive objective. Each applicable tool contributes a positive term. Same-repository or same-name tools that lack a positive judgment for a query are masked out of its in-batch negatives. This recipe uses batch size 24, two epochs, learning rate 1e-5, and temperature 0.05.

## Reproduction

Synchronize the [locked CUDA environment](PILOT_REPRODUCTION.md) before running these commands.

```sh
.venv/bin/python -m toolrank.fetch_modern --additional-shards --provenance results/mcp-source-licenses-expanded.json
.venv/bin/python -m toolrank.etom
.venv/bin/python -m toolrank.verify_sources \
  --shard .work/next/toucan.parquet \
  --shard .work/next/shards/Kimi-K2-00007.parquet \
  --shard .work/next/shards/Kimi-K2-00016.parquet \
  --shard .work/next/shards/Kimi-K2-00024.parquet \
  --shard .work/next/shards/OSS-00019.parquet \
  --shard .work/next/shards/Qwen3-00012.parquet \
  --provenance results/mcp-source-licenses-expanded.json \
  --output results/mcp-expanded-staging.json --rows .work/next/expanded-candidates.json
.venv/bin/python -m toolrank.modern_data .work/next/expanded-candidates.json --output .work/next/modern-expanded
.venv/bin/python -m toolrank.train_modern --training .work/next/modern-expanded/train.json \
  --seed 11 --output artifacts/modern-11 --record results/training/modern-11.json
```

Use `toolrank.benchmark --split etom --schema mcp` for the frozen benchmark and `--split modern-dev --corpus .work/next/modern-expanded/dev.json` for recipe selection. Select recipes using development only.

## Model selection

Compare the initial recipe with a conservative recipe (batch 32, one epoch, learning rate 5e-6) across seeds 11, 23 and 37, using `bash toolrank/modern_pilot.sh`. Select by mean development nDCG@10, breaking ties by mean development Recall@1 and then the conservative recipe. Use seed 11 of the winning recipe as the candidate. Select with the family-disjoint development corpus alone. A stronger release comparison also needs the untuned BGE baseline, which outscored ToolScout on ETOM.

The evaluation runner first verifies the exact overlap audit and executes development-only recipe selection. It calibrates abstention on the original authored development requests, which contain explicit answerable and unanswerable labels. That calibration set is small and differs from the ETOM catalog. It provides an independent threshold and a deliberately visible transfer limitation. Paired intervals resample provider clusters. Requests with equivalent tools across providers are grouped by their lexicographically first judged provider. [Overlap audit](../results/mcp-split-independence.json)
