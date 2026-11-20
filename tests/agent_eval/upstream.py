"""Serve isolated synthetic catalogs over actual MCP stdio."""
import asyncio
import sys

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from scenarios import BLOCKED, CATALOGS, RESPONSES, schema


async def run(role):
    server = Server(role)
    archived = False
    catalog = CATALOGS[role]

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=name, description=description, inputSchema=schema(argument))
                for name, (description, argument) in catalog.items() if archived or name != "lookup_archive"]

    @server.call_tool()
    async def call_tool(name, arguments):
        nonlocal archived
        tool_id = f"{role}/{name}"
        if tool_id in BLOCKED or name not in catalog or (name == "lookup_archive" and not archived):
            return types.CallToolResult(isError=True, content=[])
        value = arguments[catalog[name][1]]
        text = RESPONSES.get((tool_id, value), f"Recorded {tool_id} for {value}.")
        if tool_id == "calendar/lookup" and value == "Old Roadmap":
            archived = True
            await server.request_context.session.send_tool_list_changed()
        return [types.TextContent(type="text", text=text)]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
