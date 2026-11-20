# CPU wheel

Python 3.11–3.14. Pass local JSON tool arrays or MCP `tools/list` responses. Gateway demo and ONNX ranking passed on Linux x86-64 across all four. [Installed checks](../results/gateway_python_release.json)

```sh
uvx --python 3.13 --from 'toolrank-cpu @ https://github.com/protortyp/mcp-toolrank/releases/download/cpu-v0.6.0/toolrank_cpu-0.6.0-py3-none-any.whl' \
  toolrank-cpu catalog.json "Find museum exhibits open today" --hybrid
```

Inputs stay local. The CLI verifies all pinned INT8 files (34 MB). `--checkpoint DIR` loads trusted snapshots. `--catalog-cache DIR` caches vectors. With model and wheels cached, set `TOOLRANK_OFFLINE=1` and use `uvx --offline`.

[OpenAPI import](OPENAPI_CATALOG.md) · [Gateway](GATEWAY.md) · [CPU release](https://github.com/protortyp/mcp-toolrank/releases/tag/cpu-v0.6.0)

## PyPI publication

Install from GitHub. First PyPI upload requires a pending Trusted Publisher for `toolrank-cpu`, owner `protortyp`, repo `mcp-toolrank`, workflow `publish-cpu.yml`, environment `pypi`. [PyPI setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)

Dispatch **Publish CPU wheel to PyPI** on main with a published CPU tag. The default downloads GitHub artifacts for checksum and metadata checks. Enable `upload` after setup. Upload uses GitHub OIDC. [Workflow](../.github/workflows/publish-cpu.yml)
