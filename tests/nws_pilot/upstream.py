"""Expose NWS metadata and three allowlisted read-only pilot operations."""
import asyncio
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


BASE = "https://api.weather.gov"
ROUTES = {
    "weather__obs_station": ("/stations/KSEA", "properties"),
    "weather__alerts_types": ("/alerts/types", "eventTypes"),
    "weather__glossary": ("/glossary", "glossary"),
}
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def read_public(route, field):
    request = Request(BASE + route, headers={"User-Agent": "MCP ToolRank read-only pilot"})
    with urlopen(request, timeout=8) as response:
        if response.geturl() != BASE + route:
            raise ValueError("unexpected API response location")
        data = json.load(response)
        return {"http_status": response.status, "expected_response_field": field,
                "field_present": field in data}


async def main(catalog):
    server = Server("NWS ToolRank pilot")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=tool["name"], description=tool["description"], inputSchema=SCHEMA)
                for tool in catalog]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name not in ROUTES or arguments:
            return types.CallToolResult(isError=True, content=[])
        try:
            return await asyncio.to_thread(read_public, *ROUTES[name])
        except (OSError, ValueError):
            return types.CallToolResult(isError=True, content=[])

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main(json.loads(Path(sys.argv[1]).read_text())))
