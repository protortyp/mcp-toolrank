"""Verify explicit gateway invocation against public NWS read-only endpoints."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp.shared.memory import create_connected_server_and_client_session

from examples.openapi_catalog import catalog_from_openapi
from evaluate import fingerprint, HERE
from toolrank_cpu.gateway import build_server, Gateway, GatewayConfig


PROBES = (
    ("weather__obs_station", "Get the station metadata for KSEA, including its physical location."),
    ("weather__alerts_types", "List the available event type names for weather alerts."),
    ("weather__glossary", "Retrieve the official glossary of meteorological terms."),
)


async def run(spec, checkpoint):
    protocol = json.loads((HERE / "protocol.json").read_text())
    if fingerprint(spec) != protocol["spec_sha256"]:
        raise ValueError("OpenAPI snapshot hash differs from this pilot")
    catalog = catalog_from_openapi(json.loads(spec.read_text()), "weather")
    records = []
    with TemporaryDirectory() as directory:
        file = Path(directory) / "catalog.json"
        file.write_text(json.dumps(catalog))
        config = GatewayConfig.model_validate({"servers": [{
            "id": "nws", "command": sys.executable, "args": [str(HERE / "upstream.py"), str(file)],
            "discover": [tool["name"] for tool in catalog], "invoke": [name for name, _ in PROBES],
        }]})
        gateway = Gateway.from_config(config, checkpoint)
        async with create_connected_server_and_client_session(build_server(gateway)) as client:
            for name, text in PROBES:
                discovered = await client.call_tool("search_tools", {"query": text, "limit": 5})
                matches = discovered.structuredContent["matches"]
                tool_id = "nws/" + name
                match = next((item for item in matches if item["id"] == tool_id), None)
                record = {"id": tool_id, "present_at_five": match is not None}
                if match is not None and match["invocable"]:
                    result = await client.call_tool("invoke_tool", {
                        "tool_id": tool_id, "revision": match["revision"], "arguments": {},
                    })
                    record["error"] = bool(result.isError)
                    if not result.isError:
                        record.update(result.structuredContent)
                records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = asyncio.run(run(args.spec, args.checkpoint))
    args.output.write_text(json.dumps({"spec_sha256": fingerprint(args.spec), "probes": records}, indent=2) + "\n")
    if not all(row.get("http_status") == 200 and row.get("field_present") for row in records):
        raise SystemExit("Some live gateway probes failed")
