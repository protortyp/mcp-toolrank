# CPU wheel

The ranker uses ONNX Runtime, NumPy and Tokenizers. The staged wheel supports Python 3.12 and 3.13. Published 0.1.0 requires 3.12.

```sh
uv build --wheel cpu --out-dir dist/staged
uvx --python 3.13 --from dist/staged/toolrank_cpu-0.2.0-py3-none-any.whl \
  toolrank-cpu catalog.json "Find museum exhibits open today" --hybrid
```

The CLI downloads and verifies the pinned 34 MB INT8 checkpoint by SHA-256 and tokenizer, checks cached files, and keeps inputs local. Use `--checkpoint DIR` for trusted local weights, `--catalog-cache DIR` for vectors, and `TOOLRANK_OFFLINE=1` with `uvx --offline` after caching.

Pass a tool array or MCP `tools/list` response. See `toolrank-cpu --help` for evaluation and calibration. [OpenAPI import](OPENAPI_CATALOG.md) · [Gateway](GATEWAY.md) · [Published 0.1.0](https://github.com/protortyp/mcp-toolrank/releases/tag/cpu-v0.1.0)
