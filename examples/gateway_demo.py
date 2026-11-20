"""Run real stdio discovery against the reference MCP time and Git servers."""
import argparse
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(args):
    command = [str(args.config)]
    if args.checkpoint:
        command.extend(("--checkpoint", str(args.checkpoint)))
    async with stdio_client(StdioServerParameters(command=args.gateway, args=command)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovery = await session.call_tool("search_tools", {
                "query": "What time is it in Tokyo, and what is the Git repository status?", "limit": 10,
            })
            if discovery.isError:
                raise RuntimeError("Gateway discovery failed")
            matches = discovery.structuredContent["matches"]
            names = [match["id"] for match in matches]
            if "repository/git_status" not in names or "clock/get_current_time" not in names:
                raise RuntimeError("Both configured servers must be visible")
            executable = await session.call_tool("search_tools", {
                "query": "What time is it in Tokyo?", "limit": 1, "invocable_only": True,
            })
            if executable.isError or [row["id"] for row in executable.structuredContent["matches"]] != [
                "clock/get_current_time"
            ]:
                raise RuntimeError("The execution shortlist must contain only the allowlisted clock tool")
            tool = executable.structuredContent["matches"][0]
            response = await session.call_tool("invoke_tool", {
                "tool_id": tool["id"], "revision": tool["revision"], "arguments": {"timezone": "Asia/Tokyo"},
            })
            if response.isError:
                raise RuntimeError(response.structuredContent.get("code", "gateway invocation failed"))
            print("Discovered", ", ".join(names))
            print("Invoked", tool["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="toolrank-gateway")
    parser.add_argument("--config", type=Path, default=Path("examples/gateway_config.json"))
    parser.add_argument("--checkpoint", type=Path)
    asyncio.run(run(parser.parse_args()))
