# MCP discovery gateway

ToolRank searches allowed tools across configured stdio servers. Each result includes the original name, description, input schema, and invocation status. Stable IDs use `server/name` to distinguish duplicate names.

Run the demo from the checkout. It discovers tools on the official time and Git servers and calls the time server. Git is discovery-only.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
uv run --project cpu --no-sync python examples/gateway_demo.py
```

Configure your MCP client to run `toolrank-gateway /absolute/path/to/gateway_config.json`. Set an absolute Git repository path if the client uses a different working directory. The gateway follows paginated listings, refreshes on notifications and polling, and hides disconnected tools until reconnection.

Call `search_tools` with a query and limit. Search never executes tools. For explicit invocation, pass the match's ID, revision, and arguments to `invoke_tool`. The gateway checks the current definition, invocation allowlist, and schema. A timed-out call may already have run. Calls are never replayed.

The CPU package verifies the pinned ONNX checkpoint. Use `TOOLRANK_OFFLINE=1` with a populated cache or `--checkpoint` with a trusted snapshot. Optional `cache_dir` stores vectors keyed by model and description. Treat cosine and ranking scores as retrieval signals. Calibrate `min_cosine` with labeled requests from your catalog.

Use one gateway and cache per trusted local principal. Upstream credentials are configured per server. Arguments and responses stay out of gateway logs. Treat descriptions as untrusted data and confirm sensitive calls in the client. Clients must support the two-step interface. Local stdio trusts the client process for identity and provides no shared deployment or OAuth.

[Example configuration](../examples/gateway_config.json) · [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

Python clients can use `Gateway.from_config` and `async with gateway.running()`.
