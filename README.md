# MCP ToolRank

Shortlist English MCP and API tools by cosine similarity between a request and cached tool descriptions. The 33.36M-parameter model produces 384-dimensional embeddings and accepts up to 512 tokens.

## Benchmarks

| Model | ETOM compact nDCG@10 | ETOM compact Hit@1 | ETOM MCP nDCG@10 | ToolRet nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| MCP ToolRank 33M | 90.13% | 84.51% | 87.98% | 28.54% |
| BGE small | 88.95% | 82.25% | 87.94% | 28.81% |
| ToolScout | 81.42% | 72.30% | 80.73% | 28.66% |
| Tool-Embed-0.6B | 93.06% | 88.70% | 93.00% | 39.43% |

On 1,549 answerable ETOM requests, ToolRank ranked a valid tool first 35 more times than BGE. The +1.18-point compact nDCG@10 gain reproduced across three seeds (95% provider-cluster interval +0.53 to +1.85). CPU query encoding was 12.71 ms p50 on a Ryzen 9 5950X with four threads. [Full results](docs/RELEASE_REVIEW.md)

## Run locally

[Download the checkpoint and model card](https://huggingface.co/christianengel/mcp-toolrank-33m).

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

[CPU ONNX](examples/onnx_cpu.py) · [LangChain](examples/langchain_agent.py) · [Pydantic AI](examples/pydantic_agent.py)

## Data and limitations

Training used 5,098 Toucan requests and 263 tool descriptions matched to pinned sources. The tested compact threshold accepted 30 of 91 infeasible ETOM requests. Calibrate rejection on your catalog.

[Reproduce training and evaluation](docs/MODERN_MCP.md) · [Technical write-up](docs/MODERN_ARTICLE.md) · [Apache-2.0](LICENSE)
