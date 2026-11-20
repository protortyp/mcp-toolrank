"""Launch an extracted gateway MCPB manifest in a fresh environment."""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import platform
from subprocess import check_output
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from release_smoke import digest, TOOLS


async def check(archive, python, fresh_dependencies=False):
    with TemporaryDirectory() as temp, ZipFile(archive) as bundle:
        directory = Path(temp)
        extracted = directory / "gateway with spaces"
        if any(name.startswith("model/") for name in bundle.namelist()):
            raise ValueError("The gateway bundle should use a separate model cache")
        bundle.extractall(extracted)
        input_file = directory / "gateway configuration.json"
        input_file.write_bytes(bundle.read("example.gateway.json"))
        manifest = json.loads(bundle.read("manifest.json"))
        config = manifest["server"]["mcp_config"]
        args = [part.replace("${__dirname}", str(extracted)).replace(
            "${user_config.gateway_config}", str(input_file)) for part in config["args"]]
        cache = directory / "empty-dependencies" if fresh_dependencies else Path(
            check_output(["uv", "cache", "dir"], text=True).strip())
        if not fresh_dependencies:
            args.insert(1, "--offline")
        env = {"TOOLRANK_OFFLINE": "1", "XDG_CACHE_HOME": str(directory / "empty-cache"), "UV_PYTHON": python,
               "UV_CACHE_DIR": str(cache)}
        params = StdioServerParameters(command=config["command"], args=args, cwd=str(directory), env=env)
        started = perf_counter()
        async with asyncio.timeout(60 if fresh_dependencies else 30):
            async with stdio_client(params) as streams:
                async with ClientSession(*streams) as client:
                    await client.initialize()
                    methods = sorted(tool.name for tool in (await client.list_tools()).tools)
                    if methods != sorted(tool["name"] for tool in manifest["tools"]):
                        raise ValueError("Unexpected MCP methods")
                    search = await client.call_tool("search_tools", {
                        "query": "Read the current UTC time and count minutes since midnight", "invocable_only": True})
                    matches = {match["id"]: match["revision"] for match in search.structuredContent["matches"]}
                    if sorted(matches) != TOOLS:
                        raise ValueError("The bundled configuration must discover both tools")
                    clock = await client.call_tool("invoke_tool", {
                        "tool_id": TOOLS[0], "revision": matches[TOOLS[0]], "arguments": {}})
                    if clock.isError:
                        raise ValueError("Could not call the clock")
                    timestamp = clock.structuredContent["timestamp"]
                    calculation = await client.call_tool("invoke_tool", {
                        "tool_id": TOOLS[1], "revision": matches[TOOLS[1]], "arguments": {"timestamp": timestamp}})
                    instant = datetime.fromisoformat(timestamp)
                    if calculation.isError:
                        raise ValueError("Could not call the time calculator")
                    minutes = calculation.structuredContent["minutes"]
                    if minutes != instant.hour * 60 + instant.minute:
                        raise ValueError("The time math result differs from the clock receipt")
                    denied = await client.call_tool("invoke_tool", {
                        "tool_id": "clock/unregistered", "revision": matches[TOOLS[0]], "arguments": {}})
                    if denied.structuredContent["code"] != "not_available":
                        raise ValueError("The gateway must reject an unregistered tool")
        if (directory / "empty-cache/toolrank").exists():
            raise ValueError("The model-free example accessed the checkpoint cache")
        return {"python": python, "methods": methods, "receipt_round_trip": True,
                "unregistered": "not_available", "empty_model_cache": True,
                "wall_seconds": perf_counter() - started}


async def smoke(args):
    return {"platform": platform.platform(), "revision": check_output(["git", "rev-parse", "HEAD"],
                                                                    text=True).strip(),
            "sha256": digest(args.bundle), "bundle_bytes": args.bundle.stat().st_size,
            "offline_with_cached_dependencies": not args.fresh_dependencies,
            "checks": [await check(args.bundle, python, args.fresh_dependencies) for python in args.python],
            "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--python", nargs="+", default=["3.12", "3.13"])
    parser.add_argument("--fresh-dependencies", action="store_true", help="start with an empty uv dependency cache")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(asyncio.run(smoke(args)), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)
