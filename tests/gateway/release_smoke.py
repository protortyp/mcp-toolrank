"""Rebuild, install, and exercise local artifacts outside the checkout."""
import argparse
import asyncio
from datetime import datetime
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


TOOLS = ["clock/current_utc_time", "time_math/minutes_since_midnight"]
CATALOG = [
    {"name": "get_current_time", "description": "Read the current time in a timezone."},
    {"name": "list_exhibits", "description": "Find museum exhibits open today."},
]
CHECKPOINT_ASSETS = ("LICENSE", "model.json", "rank_onnx.py", "catalog_lexical.py", "catalog_int8.onnx")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run(command, directory):
    uv_cache = subprocess.check_output(["uv", "cache", "dir"], cwd=directory, text=True).strip()
    env = os.environ | {"TOOLRANK_OFFLINE": "1", "XDG_CACHE_HOME": str(directory / "cache"),
                        "UV_CACHE_DIR": uv_cache}
    result = subprocess.run(command, cwd=directory, env=env, check=True, capture_output=True, text=True, timeout=120)
    return result.stdout


async def desktop_bundle(bundle, directory, input_file):
    extracted = directory / "desktop"
    with ZipFile(bundle) as archive:
        notices = json.loads(archive.read("model/NOTICES.sha256.json"))
        source_manifest = Path(__file__).resolve().parents[2] / "bundle/notices.json"
        if notices != json.loads(source_manifest.read_text()):
            raise ValueError("Desktop bundle does not include the pinned model notice manifest")
        if any(sha256(archive.read(f"model/{path}")).hexdigest() != expected for path, expected in notices.items()):
            raise ValueError("Desktop bundle includes altered model notices")
        if "LICENSE" not in notices or "THIRD_PARTY/BAAI-FlagEmbedding.txt" not in notices:
            raise ValueError("Desktop bundle is missing the model or base-model license")
        archive.extractall(extracted)
    manifest = json.loads((extracted / "manifest.json").read_text())
    config = manifest["server"]["mcp_config"]
    params = [part.replace("${__dirname}", str(extracted)).replace("${user_config.catalog}", str(input_file))
              for part in config["args"]]
    params.insert(1, "--offline")
    async with stdio_client(StdioServerParameters(command=config["command"], args=params, cwd=str(directory))) as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            methods = [tool.name for tool in (await client.list_tools()).tools]
            if methods != ["search_tools"]:
                raise ValueError("Unexpected desktop bundle tools")
            result = await client.call_tool("search_tools", {"query": "What time is it now?", "limit": 1})
            matches = (result.structuredContent or {}).get("matches", [])
            if result.isError or not matches or matches[0]["tool"]["name"] != "get_current_time":
                raise ValueError("Desktop bundle returned an unexpected match")
    return {"version": manifest["version"], "search_match": "get_current_time", "model_notices_verified": len(notices)}


def inventory(args, directory):
    command = ["uvx", "--isolated", "--offline", "--python", "3.12",
               "--from", f"{args.wheel}[gateway]", "toolrank-gateway"]
    config = args.inventory_config
    if args.client_config:
        config = directory / "client import gateway.json"
        run([*command, str(args.client_config), "--import-client-config", str(config)], directory)
        imported = json.loads(config.read_text())
        if imported["ranking"] != "bm25" or any(
            row["discover"] or row["invoke"] for row in imported["servers"]
        ):
            raise ValueError("Imported config grants unexpected permissions")
    result = json.loads(run([*command, str(config), "--list-upstream-tools"], directory))["upstreams"]
    if (result["clock"]["status"] != "ready" or result["docs"]["status"] != "ready"
            or result["clock"]["tools"] != ["convert_time", "get_current_time", "get_time_ago"]
            or "search_cloudflare_documentation" not in result["docs"]["tools"]
            or (directory / "cache/toolrank").exists()):
        raise ValueError("Installed inventory could not list the reference time and docs tools")
    return result


def live_search(args, directory):
    results = {}
    for query, expected in (("Get current time", "clock/get_current_time"),
                            ("Search Cloudflare documentation", "docs/search_cloudflare_documentation")):
        start = perf_counter()
        result = json.loads(run(["uvx", "--isolated", "--offline", "--python", "3.12",
                                 "--from", f"{args.wheel}[gateway]", "toolrank-gateway", str(args.search_config),
                                 "--search", query, "--limit", "1"], directory))
        if (result["decision"] != "ranked_unverified" or result["upstreams"] != {"clock": "ready", "docs": "ready"}
                or [(row["id"], row["invocable"]) for row in result["matches"]] != [(expected, False)]
                or (directory / "cache/toolrank").exists()):
            raise ValueError("Installed one-shot search returned an unexpected result")
        results[query] = {"match": expected, "elapsed_seconds": perf_counter() - start}
    return {"queries": results, "upstreams": result["upstreams"], "empty_model_cache": True}


async def check(args):
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        release = [args.wheel, args.sdist, args.bundle]
        run(["uv", "build", "--wheel", "--offline", str(args.sdist), "--out-dir", str(directory / "rebuild")], directory)
        rebuilt = directory / "rebuild" / args.wheel.name
        if digest(rebuilt) != digest(args.wheel):
            raise ValueError("Wheel rebuilt from the isolated source distribution differs")
        with ZipFile(args.wheel) as wheel:
            if any(f"toolrank_cpu/{name}" not in wheel.namelist() for name in CHECKPOINT_ASSETS):
                raise ValueError("Wheel is missing an expected runtime asset")
        catalog = directory / "catalog.json"
        catalog.write_text(json.dumps(CATALOG))
        records = []
        for python in ("3.11", "3.12", "3.13", "3.14"):
            start = perf_counter()
            demo = json.loads(run(["uvx", "--isolated", "--offline", "--python", python,
                                   "--from", f"{args.wheel}[gateway]", "toolrank-gateway", "--demo"], directory))
            elapsed = perf_counter() - start
            utc = datetime.fromisoformat(demo["timestamp"])
            if not (demo["ranking"] == "bm25" and demo["discovered_tools"] == TOOLS == demo["invoked_tools"]
                    and demo["minutes_since_midnight"] == utc.hour * 60 + utc.minute):
                raise ValueError("Installed gateway demo returned unexpected results")
            if (directory / "cache/toolrank").exists():
                raise ValueError("The model-free demo touched the checkpoint cache")
            ranked = json.loads(run(["uvx", "--isolated", "--offline", "--python", python,
                                     "--from", str(args.wheel), "toolrank-cpu", str(catalog),
                                     "What time is it now?", "--checkpoint", str(args.checkpoint), "--hybrid",
                                     "--limit", "1"], directory))
            match = ranked[0]["tool"]["name"]
            if match != "get_current_time":
                raise ValueError(f"Installed CPU ranker returned an unexpected match on Python {python}")
            records.append({"python": python, "elapsed_seconds": elapsed, "ordered_calls": demo["invoked_tools"],
                            "minutes_match_clock_receipt": True, "empty_model_cache": True, "onnx_cli_match": match})
        upstreams = inventory(args, directory) if args.inventory_config or args.client_config else None
        search = live_search(args, directory) if args.search_config else None
        desktop = await desktop_bundle(args.bundle, directory, catalog)
    return {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": digest(Path(__file__)), "platform": platform.platform(),
            "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "artifacts": {path.name: digest(path) for path in release}, "wheel_rebuild_identical": True,
            "offline_with_cached_dependencies": True, "fresh_install_demos": records,
            "installed_onnx_cli_match": records[0]["onnx_cli_match"], "desktop_bundle": desktop,
            "installed_live_inventory": upstreams, "client_config_imported": bool(args.client_config),
            "installed_live_search": search, "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wheel", "sdist", "bundle", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    inventory_mode = parser.add_mutually_exclusive_group()
    inventory_mode.add_argument("--inventory-config", type=lambda value: Path(value).resolve(),
                                help="also inventory the reference time and Cloudflare docs servers")
    inventory_mode.add_argument("--client-config", type=lambda value: Path(value).resolve(),
                                help="import a client config for the reference time and Cloudflare docs servers")
    parser.add_argument("--search-config", type=lambda value: Path(value).resolve(),
                        help="also search the reference time and Cloudflare docs servers")
    args = parser.parse_args()
    args.output.write_text(json.dumps(asyncio.run(check(args)), indent=2) + "\n")
