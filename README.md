# MCP ToolRank

Discover, rank, and explicitly call allowlisted MCP tools over stdio or Streamable HTTP with a verified 34 MB CPU ONNX encoder.

On 11 simulated workflows, the ToolRank hybrid cut schema tokens by 85% compared with giving Qwen all tools. Both dense models and the ToolRank hybrid completed 16/22 trials. Qwen completed 12/22 with all tools. [Evaluation](docs/AGENT_EVALUATION.md)

## Try it

The demo discovers the official MCP time and Git servers, then reads the time.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
uv run --project cpu --no-sync python examples/gateway_demo.py
```

In a local MCP client, use `search_tools`, then pass the selected ID, revision, and arguments to `invoke_tool`. Set separate discovery and invocation allowlists for one trusted principal. [Configuration](examples/gateway_config.json) · [Gateway guide](docs/GATEWAY.md)

Use `toolrank-cpu` for static catalogs on Python 3.12 or 3.13. [CLI and offline setup](docs/UVX_CPU.md) · [OpenAPI import](docs/OPENAPI_CATALOG.md)

## Results and limits

ToolRank’s gain over BGE Small varies by ETOM schema under identical TF-IDF and BM25 scoring. [Benchmarks and reproduction](docs/BASELINE_COMPARISON.md)

Uncalibrated rankings leave applicability unverified. Invoke only tools permitted by the client and gateway. [Calibration](docs/ONNX_CALIBRATION.md) · [Model, provenance, and license](docs/MODERN_MODEL_CARD.md)

[LangChain](examples/langchain_agent.py) · [Pydantic AI](examples/pydantic_agent.py) · [Browser demo](demo/) · [Research history](docs/RESEARCH.md) · [Apache-2.0](LICENSE)
