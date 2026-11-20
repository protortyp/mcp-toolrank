# Gateway desktop bundle

On Linux x86-64, use an MCPB 0.4 uv-compatible host. Open the unsigned bundle and select trusted gateway JSON. The example starts two stdio providers with model-free BM25. [Gateway bundle](https://github.com/protortyp/mcp-toolrank/releases/download/gateway-mcpb-v0.3.0/mcp-toolrank-gateway-0.3.0.mcpb) · [Example](../bundle/gateway/example.gateway.json)

Stdio launches configured programs. Set separate `discover` and `invoke` allowlists for stdio or Streamable HTTP. Hybrid uses the verified checkpoint cache. First launch may install Python and dependencies.

The extracted manifest passed a two-server clock workflow on Python 3.12 and 3.13. Python 3.12 also passed with an empty dependency cache. Host GUIs remain unverified.

The builder verifies the published wheel against its lockfile.

```sh
python3 bundle/build.py dist/mcp-toolrank-gateway-0.3.0.mcpb \
  --gateway-wheel /path/to/toolrank_cpu-0.3.0-py3-none-any.whl
uv sync --project cpu --locked --extra gateway --no-editable
uv run --project cpu --no-sync python tests/gateway/gateway_bundle_smoke.py \
  dist/mcp-toolrank-gateway-0.3.0.mcpb
```

[Gateway configuration](GATEWAY.md) · [Smoke results](../results/gateway_bundle_smoke.json) · [MCP Registry](https://registry.modelcontextprotocol.io/?q=io.github.protortyp%2Fmcp-toolrank-gateway)
