"""Serve a local read-only clock over MCP Streamable HTTP."""

from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP


server = FastMCP("Loopback Clock", host="127.0.0.1", port=8271, stateless_http=True, json_response=True)


class ClockReading(TypedDict):
    timezone: str
    datetime: str


@server.tool()
def get_current_time(timezone: str) -> ClockReading:
    """Read the current time in an IANA timezone such as Asia/Tokyo."""
    timestamp = datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
    return {"timezone": timezone, "datetime": timestamp}


if __name__ == "__main__":
    server.run(transport="streamable-http")
