# Shortlisting a mixed MCP catalog

The static pilot combined 93 tools from seven providers using NWS OpenAPI and API fixtures. Twenty authored requests needed two tools. Eight were unsupported.

| System | Both tools retained / 20 | Compact tool JSON saved |
| --- | ---: | ---: |
| BM25 top ten | 18 | 91.4% |
| BGE Small INT8 hybrid top ten | 19 | 90.7% |
| ToolRank INT8 hybrid top ten | 17 | 90.4% |
| ToolRank top ten under 4,096 bytes | 16 | 93.8% |
| ToolRank whole request, top five | 14 | 95.4% |
| ToolRank supplied clauses, five slots combined | 17 | 95.4% |

The clause test used supplied splits. Every variant suggested tools for all eight unsupported requests.

Pinned Qwen tokenization counted 13,804 tokens across all projected schemas. Median schema counts were 903.5 for ToolRank top ten and 535 with the cap. Separate explicit live calls under the cap and allowlist returned HTTP 200 for NWS station and Crossref citation metadata. Other tools were discovery-only.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/run.py \
  --bge /path/to/bge-int8 --toolrank /path/to/toolrank-int8 --output /tmp/mixed.json
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/live.py \
  --checkpoint /path/to/toolrank-int8 --output /tmp/mixed-live.json
```

[Protocol](../tests/mixed_catalog/protocol.json) · [Ranking results](../results/mixed_catalog.json) · [Token counts](../results/mixed_catalog_tokens.json) · [Live results](../results/mixed_catalog_live.json)
