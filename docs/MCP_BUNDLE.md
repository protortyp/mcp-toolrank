# Desktop MCP bundle

Install uv on the client’s PATH. Open the [ToolRank 0.7.0 MCPB bundle](https://github.com/protortyp/mcp-toolrank/releases/download/v0.7.0/mcp-toolrank-0.7.0.mcpb) in an MCPB 0.4 uv-compatible host. Select a catalog JSON tool array or MCP `tools/list` response.

The existing Apache-2.0 INT8 graph powers CPU hybrid lookup. `search_tools` ranks definitions for English requests. First launch may fetch Python and dependencies. Ranking then runs offline.

Stdio on Linux x86-64/Python 3.12 passed initialization, listing, schema preservation, ranking, and threshold rejection. Host GUIs remain unverified.

Build locally with pinned model verification.

```sh
uv run --no-project --python 3.12 python bundle/build.py dist/mcp-toolrank-0.7.0.mcpb
```

Without MCPB uv support, use [MCP stdio setup](MCP_SERVER.md).
