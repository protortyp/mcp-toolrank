# MCP ToolRank

Discover, rank, and explicitly invoke allowlisted stdio/Streamable HTTP MCP tools. Model-free BM25 or hybrid with a verified 34 MB CPU encoder.

On 11 simulated workflows, the ToolRank hybrid cut schema tokens by 85% compared with giving Qwen all tools. Both dense models and the ToolRank hybrid completed 16/22 trials. Qwen completed 12/22 with all tools. [Evaluation](docs/AGENT_EVALUATION.md)

## Try it

Try the bundled offline demo from a checkout. It discovers two local tools and passes a clock reading to the calculation server.

```sh
uv build cpu --out-dir dist/staged
uvx --python 3.12 --from 'dist/staged/toolrank_cpu-0.2.0-py3-none-any.whl[gateway]' \
  toolrank-gateway --demo
```

Set separate discovery and invocation allowlists. Pass the `search_tools` result’s ID and revision with arguments to `invoke_tool`. [Configuration](examples/gateway_config.json) · [Gateway guide](docs/GATEWAY.md) · [Release preview](docs/RELEASE_PREVIEW.md)

Use `toolrank-cpu` for static catalogs on Python 3.12 or 3.13. [CLI and offline setup](docs/UVX_CPU.md) · [OpenAPI import](docs/OPENAPI_CATALOG.md)

## Results and limits

ToolRank’s gain over BGE Small varies by ETOM schema under identical TF-IDF and BM25 scoring. [Benchmarks and reproduction](docs/BASELINE_COMPARISON.md)

Uncalibrated rankings leave applicability unverified. Invoke only tools permitted by the client and gateway. [Calibration](docs/ONNX_CALIBRATION.md) · [Model, provenance, and license](docs/MODERN_MODEL_CARD.md)

[LangChain and Pydantic AI examples](docs/FRAMEWORK_INTEGRATIONS.md) · [Browser demo](demo/) · [Research history](docs/RESEARCH.md) · [Apache-2.0](LICENSE)
