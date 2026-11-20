# ToolRank CPU

`toolrank-cpu` ranks local MCP and OpenAPI catalogs from natural-language queries. The optional `gateway` extra discovers live tools with separate discovery and invocation allowlists. BM25 is model-free. Hybrid uses verified ToolRank INT8.

```sh
uvx --python 3.12 --from 'toolrank-cpu[gateway] @ https://github.com/protortyp/mcp-toolrank/releases/download/cpu-v0.6.0/toolrank_cpu-0.6.0-py3-none-any.whl' toolrank-gateway --demo
```

The clock demo uses two local MCP servers. Python 3.11–3.14 supported. [Gateway usage](https://github.com/protortyp/mcp-toolrank/blob/main/docs/GATEWAY.md) · [CPU CLI](https://github.com/protortyp/mcp-toolrank/blob/main/docs/UVX_CPU.md)
