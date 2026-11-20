# MCP ToolRank

Shortlist English MCP and API tools by cosine similarity to cached descriptions. 33.36M parameters, 384 dimensions, 512-token limit.

## Benchmarks

| Model | ETOM compact nDCG@10 | ETOM compact Hit@1 | ETOM MCP nDCG@10 | ToolRet nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| MCP ToolRank 33M | 90.13% | 84.51% | 87.98% | 28.54% |
| BGE small | 88.95% | 82.25% | 87.94% | 28.81% |
| ToolScout | 81.42% | 72.30% | 80.73% | 28.66% |
| Tool-Embed-0.6B | 93.06% | 88.70% | 93.00% | 39.43% |

Across 1,549 answerable ETOM requests, ToolRank had 35 more top-ranked hits than BGE. Compact nDCG@10 gained +1.18 points across three seeds (95% provider-cluster interval +0.53 to +1.85). CPU query encoding took 12.71 ms p50 (Ryzen 9 5950X, four threads). [Full results](docs/RELEASE_REVIEW.md)

## Run locally

[MCP stdio setup](docs/MCP_SERVER.md)

[Desktop MCP bundle](docs/MCP_BUNDLE.md) includes the existing CPU encoder for local JSON catalogs in MCPB 0.4 uv hosts (Linux x86-64 tested).

[34 MB CPU ONNX and PyTorch-free runner](docs/ONNX_INT8.md)

[Disk cache for ONNX catalog vectors](docs/ONNX_DISK_CACHE.md) reuses embeddings across CPU ranker and MCP stdio launches when catalog text matches.

[Local CPU HTTP serving](docs/TEI.md) covers ToolRank with Hugging Face TEI, prompted queries, and unprompted documents.

[OpenAPI catalog importer](docs/OPENAPI_CATALOG.md) builds one searchable operation catalog from local OpenAPI 3.0/3.1 JSON or YAML specs.

[Checkpoint and model card](https://huggingface.co/christianengel/mcp-toolrank-33m)

```sh
uv sync --locked --extra cpu --group onnx --group examples
uv run --no-sync hf download christianengel/mcp-toolrank-33m \
  --revision 08b109f86e88c43304732fcb45432dc67c294cdb \
  --local-dir artifacts/mcp-toolrank-33m
```

```python
from toolrank.selector import ToolSelector

catalog = [{"name": "list_exhibits", "description": "List exhibits open today."}]
selector = ToolSelector.from_checkpoint("artifacts/mcp-toolrank-33m", catalog)
matches = selector.rank("What can I see at the museum today?", k=3)
```

[CPU ONNX](examples/onnx_cpu.py)

[LangChain](examples/langchain_agent.py)

[Pydantic AI](examples/pydantic_agent.py)

## Data and limitations

Training used 5,098 Toucan requests and 263 pinned-source tool descriptions. The compact threshold accepted 30 of 91 infeasible ETOM requests. Calibrate on your catalog.

[Reproduce training and evaluation](docs/MODERN_MCP.md)

[Technical write-up](docs/MODERN_ARTICLE.md)

[Apache-2.0](LICENSE)
