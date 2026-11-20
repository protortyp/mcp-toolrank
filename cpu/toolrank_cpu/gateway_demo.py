"""Bundled read-only providers for the source-free gateway demo."""
import argparse
from datetime import UTC, datetime
import sys
from typing import TypedDict

from mcp.server.fastmcp import FastMCP


CLOCK = "clock/current_utc_time"
CALCULATION = "time_math/minutes_since_midnight"


class ClockReading(TypedDict):
    timestamp: str


class Minutes(TypedDict):
    minutes: int


def current_utc_time() -> ClockReading:
    """Read the current UTC time as an ISO timestamp."""
    return {"timestamp": datetime.now(UTC).isoformat(timespec="seconds")}


def minutes_since_midnight(timestamp: str) -> Minutes:
    """Count whole minutes since midnight for an ISO timestamp."""
    instant = datetime.fromisoformat(timestamp)
    return {"minutes": instant.hour * 60 + instant.minute}


async def run_demo():
    from .gateway import Gateway, GatewayConfig

    servers = []
    for tool_id in (CLOCK, CALCULATION):
        provider, _, name = tool_id.partition("/")
        servers.append({"id": provider, "command": sys.executable,
                        "args": ["-m", "toolrank_cpu.gateway_demo", provider],
                        "discover": [name], "invoke": [name]})
    config = GatewayConfig.model_validate({"ranking": "bm25", "servers": servers})
    gateway = Gateway.from_config(config)
    async with gateway.running():
        response = await gateway.search("Read current UTC time and count minutes since midnight", invocable_only=True)
        matches = {match["id"]: match["revision"] for match in response["matches"]}
        if not {CLOCK, CALCULATION} <= matches.keys():
            raise RuntimeError("Could not discover both bundled demo tools")
        clock = await gateway.invoke(CLOCK, matches[CLOCK], {})
        if clock.isError or clock.structuredContent is None:
            raise RuntimeError("Demo clock call failed")
        timestamp = clock.structuredContent["timestamp"]
        calculation = await gateway.invoke(CALCULATION, matches[CALCULATION], {"timestamp": timestamp})
        if calculation.isError or calculation.structuredContent is None:
            raise RuntimeError("Demo calculation call failed")
        return {"ranking": config.ranking, "discovered_tools": sorted(matches), "invoked_tools": [CLOCK, CALCULATION],
                "timestamp": timestamp, "minutes_since_midnight": calculation.structuredContent["minutes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("clock", "time_math"))
    args = parser.parse_args()
    server = FastMCP(args.provider)
    server.add_tool(current_utc_time if args.provider == "clock" else minutes_since_midnight)
    server.run()
