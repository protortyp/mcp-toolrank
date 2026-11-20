"""Read-only gateway tool shared by the framework examples."""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="toolrank-gateway")
    parser.add_argument("--config", type=Path, default=Path("examples/gateway_config.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18742/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("query", nargs="?", default="What time is it in Tokyo? Use the timezone Asia/Tokyo.")
    return parser.parse_args()


@asynccontextmanager
async def clock_tool(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    command = [str(args.config)]
    if args.checkpoint:
        command.extend(("--checkpoint", str(args.checkpoint)))
    async with stdio_client(StdioServerParameters(command=args.gateway, args=command)) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            found = await client.call_tool("search_tools", {"query": args.query, "limit": 3, "invocable_only": True})
            candidates = (found.structuredContent or {}).get("matches", [])
            clock = next((tool for tool in candidates if tool["id"] == "clock/get_current_time"
                          and tool["invocable"]), None)
            if clock is None:
                raise RuntimeError("The allowlisted clock tool was absent from the shortlist.")

            async def get_current_time(timezone: str) -> str:
                """Read the current time in an IANA timezone such as Asia/Tokyo."""
                observed = await client.call_tool("invoke_tool", {
                    "tool_id": clock["id"], "revision": clock["revision"], "arguments": {"timezone": timezone},
                })
                if observed.isError:
                    raise RuntimeError("The gateway clock call failed.")
                return "\n".join(item.text for item in observed.content if item.type == "text")

            yield get_current_time
