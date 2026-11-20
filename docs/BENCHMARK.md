# Benchmark report

## Decision

**NO-GO for release or paid training on this evidence.** On the small unseen-family test, the candidate queued the correct tool first for 32 of 36 answerable requests, versus 30 for ToolScout and 29 for BGE. The 95% paired family-bootstrap interval for the candidate's +5.56 percentage-point difference against ToolScout spans 0.00 to +16.67 points. The sample covers four families. The candidate scored 26.62% nDCG@10 on ToolRet versus 28.66% for ToolScout and 28.81% for BGE. It missed the 2-point tolerance versus ToolScout by 0.03 points and has effectively the same CPU latency and model size. Tool-Embed reached 39.43% nDCG@10 with a much larger model.

## Measurement conditions

All runs used the local Ryzen 9 5950X and RTX 3090. ToolRet quality is based on exact cosine search over 44,453 tools for 7,961 requests across 35 tasks, with zero missing judged tool IDs. Recall counts positive tools retrieved per request. nDCG@10 uses linear graded gains and a per-query ideal ranking. Scores are query means. The system used the downloadable release and original documentation, omitted benchmark-supplied task instructions, and used model-specific query prefixes. The public run allows 512 tokens for BGE, ToolScout and Tool-Embed, and uses the candidate's saved limit of 128. This protocol differs from author reported category averages, the MTEB task, and expanded Tool-REX documents.

Encoder profiles use FP32, four CPU threads, and a fixed eight-query workload distinct from model training and retrieval evaluation. Batch-one latency includes tokenization, device synchronization on CUDA, and normalization but excludes search and model load. Batch-16 throughput uses a warmed batch workload. CPU memory is peak RSS for the encoder process including imports and model load. GPU memory is peak PyTorch allocated device memory in the encoder profile. The GPU was shared with other local workloads. Full-catalog document throughput comes from one run per model. Treat the GPU timings as descriptive.

The unseen-family test contains 36 answerable and 12 unanswerable requests across four unseen families and 12 tools. Each fixed schema variant uses a threshold calibrated on the separate development set. Coverage is the share of answerable queries for which the top score reaches that threshold. None false accepts is the share of unanswerable queries with a top score that reaches it. Data and protocols are recorded in the [experiment plan](EXPERIMENT.md) and [split audit](../results/audit.json).

## Errors on the authored holdout

- The candidate treats an inspection report about a cracked object as completed conservation work. Both tools describe records of damage or stabilization, but the request calls for documenting a finding.
- It routes an already completed school pickup to the approval tool on one query and to an unrelated beekeeping tool on another. The catalog distinguishes recording a completed action from authorizing one.
- It routes a bent sewing needle to conservation treatment. The tool description and query share a repair intent without the correct entity match.
- The candidate rejects 9 of 12 unanswerable requests with its plain-text threshold. The false accepts include requests for appraisal, grades, and photo-based insect identification. Its nearest available tools share a domain with those requests. [Per-query results](../results/test/candidate-plain-rows.json)

## Errors on ToolRet

The 128-token cap truncates 28,036 of 44,453 tool descriptions. On the 5,283 requests with at least one positive tool above the limit, candidate nDCG@10 is 23.22% versus ToolScout's 26.94%. On the remaining 2,678 requests it is 33.34% versus 32.04%. The checkpoint and serving length differ, so this grouping cannot establish that truncation caused the change. The largest candidate decline by task versus ToolScout was UltraTool, from 18.31% to 4.83% nDCG@10. [Length diagnostic](../results/toolret-diagnostics.json)

## Scope and reproduction

Baseline and candidate weights, data revisions, and environment are pinned. The pilot varies recipes jointly and cannot attribute the development ranking to any particular hyperparameter. The test has one positive judgment per answerable request, little catalog ambiguity outside same-family tools, and small sample sizes for rejection estimates. Public checkpoint pretraining overlap remains unverified. Use the [reproduction commands](PILOT_REPRODUCTION.md), [seed records](../results/training), and [paired analysis](../results/paired.json) to inspect the experiment.

A second seed-11 training run produced the same loss trace and identical safetensors SHA-256. An independent NumPy ranking replay recovered the same held-out top-one counts for every model and schema. [Training replay](../results/reproduction.json) · [Ranking replay](../results/holdout_replay.json)
## ToolRet full catalog

| Model | Parameters | R@1 | R@5 | R@10 | nDCG@10 | RTX docs/s |
| --- | --- | --- | --- | --- | --- | --- |
| BGE small | 33.36M | 16.51% | 31.11% | 37.12% | 28.81% | 751 |
| ToolScout | 33.36M | 16.31% | 31.11% | 36.97% | 28.66% | 391 |
| Tool-Embed-0.6B | 595.78M | 22.09% | 42.33% | 50.58% | 39.43% | 53 |
| Candidate | 33.36M | 15.90% | 28.75% | 34.13% | 26.62% | 577 |

## CPU encoder profile

| Model | p50 ms | p95 ms | Batch-16 queries/s | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| BGE small | 12.35 | 14.42 | 218 | 985 |
| ToolScout | 12.71 | 14.47 | 224 | 985 |
| Tool-Embed-0.6B | 192.64 | 212.47 | 10 | 4268 |
| Candidate | 12.75 | 14.44 | 207 | 1010 |

## RTX 3090 encoder profile

| Model | p50 ms | p95 ms | Batch-16 queries/s | Peak CUDA MiB |
| --- | --- | --- | --- | --- |
| BGE small | 9.60 | 12.21 | 1471 | 143 |
| ToolScout | 9.48 | 12.87 | 1581 | 143 |
| Tool-Embed-0.6B | 23.99 | 28.07 | 305 | 2318 |
| Candidate | 7.87 | 9.50 | 2264 | 143 |

## Unseen-family holdout

| Model | Plain R@1 | MCP R@1 | OpenAI R@1 | Plain R@5 | Plain R@10 | Plain nDCG@10 | Plain coverage | Plain none false accepts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BGE small | 80.56% | 83.33% | 83.33% | 97.22% | 100.00% | 91.33% | 100.00% | 83.33% |
| ToolScout | 83.33% | 83.33% | 86.11% | 97.22% | 100.00% | 91.25% | 91.67% | 58.33% |
| Tool-Embed-0.6B | 75.00% | 83.33% | 77.78% | 100.00% | 100.00% | 90.22% | 91.67% | 16.67% |
| Candidate | 88.89% | 88.89% | 88.89% | 97.22% | 100.00% | 94.41% | 94.44% | 25.00% |
