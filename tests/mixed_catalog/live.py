"""Test a capped execution shortlist through the real gateway and two public APIs."""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp.shared.memory import create_connected_server_and_client_session
from toolrank_cpu.gateway import Gateway, GatewayConfig, build_server

from run import HERE, catalog, schema_bytes
from upstream import READS


PROBES = (
    ("nws", "weather__obs_station", "Find the recorded location of surface observing station KSEA."),
    ("crossref", "crossref_work_by_doi", "Get citation metadata for DOI 10.1038/nature14539."),
)
CAP = 4096


async def run(checkpoint):
    entries = catalog()
    providers = defaultdict(list)
    for entry in entries:
        providers[entry.server].append(entry.tool.model_dump(exclude_none=True))
    reports = []
    with TemporaryDirectory() as temporary:
        servers = []
        for provider, tools in sorted(providers.items()):
            file = Path(temporary) / f"{provider}.json"
            file.write_text(json.dumps(tools))
            servers.append({
                "id": provider, "command": sys.executable, "args": [str(HERE / "upstream.py"), provider, str(file)],
                "discover": [tool["name"] for tool in tools], "invoke": [name for owner, name in READS if owner == provider],
            })
        gateway = Gateway.from_config(GatewayConfig.model_validate({"servers": servers}), checkpoint)
        async with create_connected_server_and_client_session(build_server(gateway)) as client:
            limited = await client.call_tool("search_tools", {
                "query": PROBES[0][2], "limit": 10, "invocable_only": True, "max_tool_bytes": 2,
            })
            if limited.isError or limited.structuredContent["matches"] or (
                limited.structuredContent["decision"] != "max_tool_bytes_exceeded"
            ):
                raise RuntimeError("Gateway did not report a budget-limited empty shortlist")
            for provider, tool, query in PROBES:
                discovery = await client.call_tool("search_tools", {
                    "query": query, "limit": 10, "invocable_only": True, "max_tool_bytes": CAP,
                })
                if discovery.isError or discovery.structuredContent is None:
                    raise RuntimeError("Discovery failed")
                matches = discovery.structuredContent["matches"]
                tool_id = f"{provider}/{tool}"
                match = next((row for row in matches if row["id"] == tool_id), None)
                size = schema_bytes(matches)
                if not match or size > CAP or any(not row["invocable"] for row in matches):
                    raise RuntimeError("Gateway omitted the known safe operation or violated its shortlist scope")
                response = await client.call_tool("invoke_tool", {
                    "tool_id": tool_id, "revision": match["revision"], "arguments": READS[provider, tool][0],
                })
                reports.append({"tool": tool_id, "matches_bytes": size, "shortlist": len(matches),
                                "invocation_error": bool(response.isError), **(response.structuredContent or {})})
    return {"catalog_tools": len(entries), "catalog_providers": len(providers), "max_tool_bytes": CAP,
            "empty_budget_decision": limited.structuredContent["decision"], "probes": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.checkpoint))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not all(row.get("http_status") == 200 and row.get("requested_field_present") for row in result["probes"]):
        raise SystemExit("Live gateway read probe failed")
