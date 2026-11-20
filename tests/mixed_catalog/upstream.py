"""Serve benchmark schemas and two fixed public read operations over MCP."""
import asyncio
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


READS = {
    ("nws", "weather__obs_station"): (
        {"stationId": "KSEA"}, "https://api.weather.gov/stations/KSEA", "properties"
    ),
    ("crossref", "crossref_work_by_doi"): (
        {"doi": "10.1038/nature14539"}, "https://api.crossref.org/works/10.1038/nature14539", "message"
    ),
}


def fetch(url, field):
    request = Request(url, headers={"User-Agent": "MCP ToolRank public-read mixed catalog pilot"})
    with urlopen(request, timeout=8) as response:
        if response.geturl() != url:
            raise ValueError("unexpected response location")
        content = json.load(response)
        return {"http_status": response.status, "requested_field_present": field in content}


async def run(server_name, tools):
    server = Server(server_name)

    @server.list_tools()
    async def list_tools():
        return [types.Tool.model_validate(tool) for tool in tools]

    @server.call_tool()
    async def call_tool(name, arguments):
        mapping = READS.get((server_name, name))
        if not mapping or arguments != mapping[0]:
            return types.CallToolResult(isError=True, content=[])
        try:
            return await asyncio.to_thread(fetch, *mapping[1:])
        except (ValueError, OSError):
            return types.CallToolResult(isError=True, content=[])

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], json.loads(Path(sys.argv[2]).read_text())))
