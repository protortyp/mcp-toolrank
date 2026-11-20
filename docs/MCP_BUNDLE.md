# Desktop MCP bundle

Install uv on the client’s PATH. Download the [ToolRank 0.6.0 MCPB bundle](https://github.com/protortyp/mcp-toolrank/releases/download/v0.6.0/mcp-toolrank-0.6.0.mcpb) and open it in a host supporting MCPB 0.4 uv servers. When prompted, select your catalog JSON tool array or MCP `tools/list` response.

The bundle includes the existing Apache-2.0 INT8 graph. `search_tools` performs lookup only, returning ranked definitions for English requests via CPU hybrid retrieval. First launch may fetch Python and dependencies. Ranking runs offline after setup.

We verified stdio initialization, tool listing, schema preservation, ranking, and threshold rejection on Linux x86-64 with Python 3.12. Host GUIs remain unverified.

Build the bundle from this checkout. The script verifies pinned model files.

```sh
uv run --no-project --python 3.12 python bundle/build.py dist/mcp-toolrank-0.6.0.mcpb
```

Without MCPB uv support, use [MCP stdio setup](MCP_SERVER.md).
