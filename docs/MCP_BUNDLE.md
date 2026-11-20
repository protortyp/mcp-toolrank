# Desktop catalog search

Install uv on PATH. Open [MCPB 0.7.1 bundle](https://github.com/protortyp/mcp-toolrank/releases/download/cpu-v0.2.0/mcp-toolrank-0.7.1.mcpb) in a uv-compatible MCPB 0.4 host. Select a JSON tool array or MCP `tools/list` response.

`search_tools` ranks English queries on CPU with INT8 weights and notices. First launch may install Python and dependencies. Linux x86-64/Python 3.12 stdio verified. Host GUIs remain unverified.

```sh
uv run --no-project --python 3.12 python bundle/build.py dist/mcp-toolrank-0.7.1.mcpb
```

Use the [Gateway](GATEWAY.md) for live discovery and invocation. [MCP Registry](https://registry.modelcontextprotocol.io/?q=io.github.protortyp%2Fmcp-toolrank)
