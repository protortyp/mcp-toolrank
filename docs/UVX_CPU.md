# Run ToolRank with uvx

Use uv and Python 3.12 from a clean directory. The PyTorch-free CPU wheel uses ONNX Runtime, NumPy, and Tokenizers.

```sh
uvx --python 3.12 \
  --from https://github.com/protortyp/mcp-toolrank/releases/download/cpu-v0.1.0/toolrank_cpu-0.1.0-py3-none-any.whl \
  toolrank-cpu catalog.json "Find museum exhibits open today" --hybrid
```

First ranking fetches the existing pinned 34 MB INT8 model and tokenizer from Hugging Face. SHA-256 checks cover downloads and local cache reuse. Requests and catalogs stay local.

Use `--checkpoint DIR` for a compatible ONNX snapshot. After setup, set `TOOLRANK_OFFLINE=1` to require verified cached model files and add uvx `--offline` for cached dependencies. `--catalog-cache DIR` persists document embeddings across launches.

Pass a tool array or MCP `tools/list` response. Convert OpenAPI JSON or YAML with the [catalog importer](OPENAPI_CATALOG.md). `--help` lists evaluation and calibration options.

Build from this checkout with `uv build --wheel cpu --out-dir dist/cpu`.
