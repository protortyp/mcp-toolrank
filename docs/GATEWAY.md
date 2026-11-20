# MCP discovery gateway

ToolRank searches allowlisted tools over stdio and Streamable HTTP. Results preserve names, descriptions, input schemas, and invocation status. Stable `server/name` IDs distinguish duplicates.

Run the demo from the checkout. It discovers tools on the official time and Git servers and calls the time server. Git is discovery-only.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
uv run --project cpu --no-sync python examples/gateway_demo.py
```

Configure your MCP client to run `toolrank-gateway /absolute/path/to/gateway_config.json`. Set an absolute Git repository path if the client uses a different working directory. The gateway follows paginated listings, refreshes on notifications and polling, and hides disconnected tools until reconnection.

Call `search_tools` with a query and limit. `max_tool_bytes` caps the compact UTF-8 `matches` array to a ranked prefix. An oversized first match returns `max_tool_bytes_exceeded`. Set `invocable_only` to true to apply the invocation allowlist before shortlisting. Search never executes tools. For explicit invocation, pass the match's ID, revision, and arguments to `invoke_tool`. The gateway checks the current definition, invocation allowlist, and schema. A timed-out call may already have run. Calls are never replayed.

The CPU package verifies the pinned ONNX checkpoint. Use `TOOLRANK_OFFLINE=1` with a populated cache or `--checkpoint` with a trusted snapshot. Optional `cache_dir` stores vectors keyed by model and description. Treat cosine and ranking scores as retrieval signals. Calibrate `min_cosine` with labeled requests from your catalog.

Use one gateway and cache per trusted local principal. Upstream credentials are configured per server. Arguments and responses stay out of gateway logs. Treat descriptions as untrusted data and confirm sensitive calls in the client. Clients must support the two-step interface. Local stdio trusts the client process for identity and provides no shared deployment or OAuth.

[Example configuration](../examples/gateway_config.json) · [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

Python clients can use `Gateway.from_config` and `async with gateway.running()`.

## Model-free ranking

Set `"ranking": "bm25"` for lexical search without a checkpoint or ONNX session. Hybrid is the default and supports `min_cosine`. BM25 matches have `cosine: null`. `no_lexical_overlap` means eligible tools share no indexed query terms. Overlap leaves relevance unverified. [Measured trade-off](GATEWAY_RANKING.md)

## Streamable HTTP upstreams

Mix stdio and HTTP entries. For HTTP, set `transport` to `streamable_http` and `url` to the MCP endpoint. Optional `bearer_token_env` names a per-upstream token variable. Discovery and invocation allowlists apply. Remote endpoints require HTTPS. Plain HTTP is loopback-only. Direct connections reject redirects to prevent invocation replay. Clients use stdio. The pinned SDK negotiates through MCP `2025-11-25`.

With the gateway extra installed, use separate terminals. The demo calls the HTTP clock and keeps Git discovery-only.

```sh
uv run --project cpu --no-sync python examples/gateway_http_clock.py
uv run --project cpu --no-sync python examples/gateway_demo.py \
  --config examples/gateway_http_config.json
```

The installed wheel discovered `search_cloudflare_documentation` at `https://docs.mcp.cloudflare.com/mcp` with discovery-only permissions. [Mixed configuration](../examples/gateway_http_config.json) · [Smoke results](../results/gateway_http_smoke.json) · [MCP transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) · [Cloudflare endpoint](https://github.com/cloudflare/mcp-server-cloudflare/blob/main/server.json)

## MCP Inspector

Run from the checkout with an installed CPU wheel and local INT8 checkpoint. The test pins Inspector 2.16.0 and Node 24.22.1. It checks both gateway methods over stdio, discovers five tools across the reference time and Git servers, invokes the allowlisted clock after invocation-only search, and verifies `not_available` for the discovery-only Git tool.

```sh
python3 tests/gateway/inspector_smoke.py \
  --gateway /path/to/toolrank-gateway --checkpoint /path/to/toolrank-int8
```

[Inspector CLI](https://github.com/modelcontextprotocol/inspector/blob/main/clients/cli/README.md) · [Smoke results](../results/gateway_inspector.json)
