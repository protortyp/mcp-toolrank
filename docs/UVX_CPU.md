# CPU wheel

Python 3.12 or 3.13. Pass JSON tool arrays or MCP `tools/list` responses.

```sh
uvx --python 3.13 --from 'toolrank-cpu @ https://github.com/protortyp/mcp-toolrank/releases/download/cpu-v0.3.0/toolrank_cpu-0.3.0-py3-none-any.whl' \
  toolrank-cpu catalog.json "Find museum exhibits open today" --hybrid
```

Inputs stay local. The CLI verifies all pinned INT8 files (34 MB). `--checkpoint DIR` loads trusted snapshots. `--catalog-cache DIR` caches vectors. With model and wheels cached, set `TOOLRANK_OFFLINE=1` and use `uvx --offline`.

[OpenAPI import](OPENAPI_CATALOG.md) · [Gateway](GATEWAY.md) · [CPU release](https://github.com/protortyp/mcp-toolrank/releases/tag/cpu-v0.3.0)
