---
library_name: sentence-transformers
base_model: BAAI/bge-small-en-v1.5
language: en
pipeline_tag: sentence-similarity
tags:
- tool-retrieval
- mcp
---

# MCP tool retrieval candidate

**Local research checkpoint. Distribution awaits the project's release decision and final approval.**

This candidate embeds English natural-language requests and static API or MCP tool descriptions. It is a 33,360,000-parameter SentenceTransformer with 384-dimensional normalized output, BGE CLS pooling, and a 128-token limit. It was initialized from the MIT-tagged BGE small revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.

## Intended use

Preselect a small set of tool definitions from a caller's existing, authorized catalog. Use `encode_query` for requests and `encode_document` for tool names and descriptions. Use cosine or dot product on normalized embeddings. The [selector example](../README.md#run-locally) preserves the original tool schemas for subsequent use. Scores are similarities. Calibrate any abstention threshold for the model, schema representation, and target catalog.

## Data and training

The candidate was trained with 88 original single-intent requests paired with 44 fictional tools from 11 families. The selected recipe uses in-batch InfoNCE, AdamW, four epochs, learning rate 2e-5, batch 16, temperature 0.05, and seed 11. The three recipe groups and nine runs are recorded in [the experiment plan](EXPERIMENT.md) and [run records](../results/training). Model selection only used the disjoint development set. The held-out data did not enter the optimizer, mining, development thresholds, or example generation.

See [the research log](RESEARCH.md) for licenses and inherited provenance. The base is tagged MIT. The candidate is built from the base plus project-authored examples. Proposed release terms for new project material require owner approval. The local package includes the MIT notice from the upstream FlagEmbedding project. Public benchmark corpora are acquired separately for local scoring.

## Evaluation and limitations

See [the benchmark report](BENCHMARK.md) for exact ToolRet, unseen-family, abstention, and hardware results, and [the raw measurements](../results) for reproduction. The authored holdout is small and uses 12 concise fictional tools. Its JSON variants exercise field layout with minimal argument metadata. ToolRet covers more diverse descriptions and positive sets, but the 128-token cap can truncate important text. Neither benchmark measures downstream tool execution, robust argument inference, multilingual retrieval, tool sequences, or production abstention behavior. Public pretraining provenance does not rule out benchmark exposure by the base or specialist comparison models.

The package contains a PyTorch SentenceTransformer and a CPU ONNX Runtime graph. The exported graph passed six PyTorch parity cases with maximum absolute embedding error of 1.49e-7. See [the repository instructions](../README.md) and [the exact export record](../results/onnx.json).

Candidate ToolRet Recall@1/5/10 was 15.90% / 28.75% / 34.13%, with 26.62% nDCG@10 across 7,961 queries. The small authored test yielded 88.89% Recall@1 across all three schema formats. The plain representation reached 94.44% answerable coverage and 25.00% false accepts on unanswerable requests. In the local encoder profile, CPU p50/p95 were 12.75/14.44 ms and the RTX 3090 p50/p95 were 7.87/9.50 ms. Exact unrounded metrics are in the raw results.
