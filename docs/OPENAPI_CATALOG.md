# Search OpenAPI operations

Import local OpenAPI 3.0 or 3.1 JSON or YAML into one searchable catalog. Each `--api` takes a namespace and path. Use distinct namespaces to keep shared operation IDs separate.

```sh
uv run --locked --script examples/openapi_catalog.py \
  --api museum ./museum.yaml --api tickets ./tickets.json > catalog.json

uv run --locked --script examples/rank_onnx.py \
  artifacts/toolrank-int8 ./catalog.json "Find museum exhibits open today" --hybrid
```

Get `artifacts/toolrank-int8` via [CPU ONNX setup](ONNX_INT8.md). The catalog also works with the [desktop bundle](MCP_BUNDLE.md) and [MCP stdio server](MCP_SERVER.md).

Search text combines operation summaries, descriptions, HTTP methods, and paths. `_meta["toolrank/openapi"]` preserves the original operation ID, method, path, and namespace. Missing IDs get route-derived names.

The importer reads `paths` without network access. It resolves local path item references and rejects external path item references and duplicate catalog names. Results support lookup only and omit executable argument schemas.
