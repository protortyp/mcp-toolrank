"""Compute the minute of day from an explicit timestamp over MCP stdio."""
from datetime import datetime
from typing import TypedDict

from mcp.server.fastmcp import FastMCP


server = FastMCP("Time Math")


class MinuteOfDay(TypedDict):
    minutes: int


@server.tool()
def minutes_since_midnight(timestamp: str) -> MinuteOfDay:
    """Count whole minutes from midnight using the local time in an ISO timestamp."""
    instant = datetime.fromisoformat(timestamp)
    return {"minutes": instant.hour * 60 + instant.minute}


if __name__ == "__main__":
    server.run()
