---
library_name: sentence-transformers
base_model: BAAI/bge-small-en-v1.5
language: en
pipeline_tag: sentence-similarity
license: apache-2.0
tags:
- tool-retrieval
- mcp
---

# MCP tool retriever, 33.36M research candidate

This local review artifact embeds English requests and tool definitions for cosine retrieval. It produces 384-dimensional embeddings with a 512-token limit and is intended to shortlist tools from a caller-authorized catalog. The selector uses compact `name: description` text. The evaluator also supports real MCP and OpenAI JSON input schemas. Tool invocation and permission checks belong to the calling application.

## Provenance and configuration

The base is [BGE small at its pinned MIT-tagged revision](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a). The training split uses 5,098 requests and 263 documented tools, built from the pinned Apache-2.0-tagged Toucan release and descriptions matched to versioned MIT source files. The package includes 39 source-repository notices and the base and query-data notices. The [license and source manifest](../results/mcp-expanded-staging.json) identifies exact repository commits, files and license blobs. Source snapshots and query excerpts stay local.

Development selected seed 11 of the conservative recipe using three-seed mean nDCG@10. It uses batch size 32, one epoch, learning rate 5e-6, a multi-positive contrastive objective, and BF16 autocast on an RTX 3090. The saved model runs in FP32. Family-disjoint development selected the training recipe, a separate authored development set calibrated abstention, and the frozen ETOM scoring data was used for evaluation. [Training record](../results/training/modern-conservative-11.json) · [Selection](../results/modern-selection.json) · [Overlap audit](../results/mcp-split-independence.json)

## Measured use and limitations

On ETOM with compact descriptions, the candidate reached 90.13% nDCG@10 and 84.51% Hit@1. The matching BGE baseline reached 88.95% and 82.25%. Paired nDCG@10 improvement was 1.18 percentage points with a 95% provider-cluster interval of 0.53 to 1.85 points. Native MCP JSON reached 87.98% nDCG@10, effectively tied with BGE and above ToolScout's 80.73%. The larger Tool-Embed model scored 93.00% with native MCP JSON. The candidate encoded queries at 12.71 ms median on the tested four-thread CPU setup. [ETOM MCP](../results/etom/candidate-mcp.json) · [ETOM plain](../results/etom/candidate-plain.json) · [Plain paired comparison](../results/etom/plain-paired.json) · [CPU runtime](../results/modern-runtime/candidate-cpu.json)

At the independently calibrated plain-text threshold, roughly one in three infeasible ETOM requests was accepted. Native MCP JSON accepted over six in ten. Use retrieval scores to rank available tools and validate task feasibility before acting. Reported results cover English retrieval, not argument generation, tool execution or end-to-end task success. Catalog and schema choices affect accuracy and score calibration. The ETOM registry is kept as a separate local evaluation input.

The SentenceTransformer and CPU ONNX graph are both available in the local checkpoint. [ONNX verification](../results/modern-onnx.json) · [Reproduction](MODERN_MCP.md)

ToolRet reached 28.54% nDCG@10 and Recall@1/5/10 of 16.36% / 31.21% / 36.91%. The two other fixed seeds reached 90.24% and 90.36% nDCG@10 on ETOM with compact descriptions. [ToolRet](../results/modern-toolret.json) · [Two-seed replication and runtime](RELEASE_REVIEW.md)
