"""Expose schemas and deterministic read receipts over MCP stdio."""
import asyncio
import json
from pathlib import Path
import sys

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


async def run(provider, fixture):
    server = Server(provider)
    receipts = {row["key"]: row["receipt"] for row in fixture["receipts"]}

    @server.list_tools()
    async def list_tools():
        return [types.Tool.model_validate(tool) for tool in fixture["tools"]]

    @server.call_tool()
    async def call_tool(name, arguments):
        key = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if key in receipts:
            return receipts[key]
        return types.CallToolResult(isError=True, content=[])

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], json.loads(Path(sys.argv[2]).read_text())))
