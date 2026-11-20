# Use ToolRank as an MCP server

The read-only stdio server caches tool embeddings at startup and returns matching definitions with original input schemas.

## Configure a client

Clone the repository, [download the INT8 model and tokenizer](ONNX_INT8.md), and export `catalog.json` as a tools array, MCP `tools/list` result, or full JSON-RPC response. Keep `search_mcp.py`, its uv lockfile, and `rank_onnx.py` together. Substitute absolute paths for clients that accept `mcpServers`.

```json
{
  "mcpServers": {
    "toolrank": {
      "command": "uv",
      "args": [
        "run", "--locked", "--script", "/path/to/mcp-toolrank/examples/search_mcp.py",
        "/path/to/toolrank-int8", "/path/to/catalog.json"
      ]
    }
  }
}
```

Call `search_tools` with a natural-language `query`, optional `limit` (default 5), and optional `min_score` (calibrate the cosine threshold on your catalog). The structured result contains a `matches` array of `tool` definitions and their cosine `score` values.

The CPU server runs without PyTorch. Restart it after catalog changes. Run tools through separate connections to their providers.
