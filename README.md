# MCP ToolRank

Discover, rank, and explicitly invoke allowlisted stdio/Streamable HTTP MCP tools. Model-free BM25 or hybrid with a verified 34 MB CPU encoder.

On 11 simulated workflows, the ToolRank hybrid cut schema tokens by 85% compared with giving Qwen all tools. Both dense models and the ToolRank hybrid completed 16/22 trials. Qwen completed 12/22 with all tools. [Evaluation](docs/AGENT_EVALUATION.md)

## Try it

The demo passes a clock reading between two local servers. It runs without a model download.

```sh
uvx --python 3.12 --from 'toolrank-cpu[gateway] @ https://github.com/protortyp/mcp-toolrank/releases/download/cpu-v0.5.0/toolrank_cpu-0.5.0-py3-none-any.whl' toolrank-gateway --demo
```

Import servers, inventory tools, and grant discovery and invocation separately. Search live from the shell. Pass the `search_tools` ID and revision to `invoke_tool`. [CLI search](docs/GATEWAY.md#search-from-the-shell) · [Import and inventory](docs/GATEWAY.md#import-an-mcp-client-config) · [Configuration](examples/gateway_config.json) · [Gateway guide](docs/GATEWAY.md) · [CPU release](https://github.com/protortyp/mcp-toolrank/releases/tag/cpu-v0.5.0) · [Desktop bundle](docs/GATEWAY_BUNDLE.md)

Use `toolrank-cpu` for static catalogs on Python 3.12 or 3.13. [CLI and offline setup](docs/UVX_CPU.md) · [OpenAPI import](docs/OPENAPI_CATALOG.md)

## Results and limits

ToolRank’s gain over BGE Small varies by ETOM schema under identical TF-IDF and BM25 scoring. [Benchmarks and reproduction](docs/BASELINE_COMPARISON.md)

Uncalibrated rankings leave applicability unverified. Invoke only tools permitted by the client and gateway. [Calibration](docs/ONNX_CALIBRATION.md) · [Model, provenance, and license](docs/MODERN_MODEL_CARD.md)

[LangChain and Pydantic AI examples](docs/FRAMEWORK_INTEGRATIONS.md) · [Browser demo](demo/) · [Research history](docs/RESEARCH.md) · [Apache-2.0](LICENSE)
